import os
from typing import Generator, Tuple, Optional
import time
import tqdm
import torch
from loguru import logger
from transformers import AutoModelForCausalLM
from kimia_infer.models.detokenizer import get_audio_detokenizer
from .prompt_manager import KimiAPromptManager
from kimia_infer.utils.sampler import KimiASampler
from huggingface_hub import snapshot_download

# Configure loguru logger to write to file
logger.add("kimia_audio.log", rotation="10 MB", level="DEBUG")  # Add file handler with rotation
class KimiAudio(object):
    def __init__(self, model_path: str, load_detokenizer: bool = True):
        logger.info(f"Loading kimi-audio main model")

        if os.path.exists(model_path):
            # local path
            cache_path = model_path
        else:
            # cache everything if model_path is a model-id
            cache_path = snapshot_download(model_path)
    
        logger.info(f"Looking for resources in {cache_path}")
        logger.info(f"Loading whisper model")
        self.alm = AutoModelForCausalLM.from_pretrained(
            cache_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        self.alm = self.alm.to(torch.cuda.current_device())

        model_config = self.alm.config
        self.kimia_token_offset = model_config.kimia_token_offset

        self.prompt_manager = KimiAPromptManager(
            model_path=cache_path, kimia_token_offset=self.kimia_token_offset
        )

        if load_detokenizer:
            logger.info(f"Loading detokenizer")
            # need to compile extension moudules for the first time, it may take several minutes.
            self.detokenizer = get_audio_detokenizer(cache_path)
        else:
            # in this case, you're not allowed to generate audio(wav)
            self.detokenizer = None

        self.extra_tokens = self.prompt_manager.extra_tokens
        self.kimia_text_audiodelaytokens = 6
        self.eod_ids = [self.extra_tokens.msg_end, self.extra_tokens.media_end]
        
        # Streaming parameters
        self.stream_chunk_size = 20  # Number of audio tokens to generate before streaming
        self.audio_chunk_size = 30   # Chunk size for audio detokenization

    @torch.inference_mode()
    def _generate_loop(
        self,
        audio_input_ids: torch.Tensor,  # input audio tokens
        text_input_ids: torch.Tensor = None,  # input text tokens if use multi-input
        max_new_tokens: int = 50,
        audio_top_k: int = 5,
        audio_temperature: float = 0.0,
        audio_repetition_penalty: float = 1.0,
        audio_repetition_window_size: int = 64,
        text_top_k: int = 5,
        text_temperature: float = 0.0,
        text_repetition_penalty: float = 1.0,
        text_repetition_window_size: int = 16,
        is_continuous_mask: torch.Tensor = None,
        continous_feature: torch.Tensor = None,
        output_type: str = "text",
    ):

        sampler = KimiASampler(
            audio_top_k=audio_top_k,
            audio_temperature=audio_temperature,
            audio_repetition_penalty=audio_repetition_penalty,
            audio_repetition_window_size=audio_repetition_window_size,
            text_top_k=text_top_k,
            text_temperature=text_temperature,
            text_repetition_penalty=text_repetition_penalty,
            text_repetition_window_size=text_repetition_window_size,
        )

        text_stream_is_finished = False
        previous_audio_tokens = torch.zeros(
            (4096,),
            dtype=torch.int,
            device=torch.cuda.current_device(),
        )
        text_previous_tokens = torch.zeros(
            (4096,),
            dtype=torch.int,
            device=torch.cuda.current_device(),
        )

        decoder_input_audio_ids = audio_input_ids.clone()
        decoder_input_text_ids = text_input_ids.clone()
        decoder_position_ids = (
            torch.arange(
                0, decoder_input_audio_ids.shape[1], device=torch.cuda.current_device()
            )
            .unsqueeze(0)
            .long()
        )
        decoder_input_whisper_feature = continous_feature
        decoder_is_continuous_mask = is_continuous_mask
        past_key_values = None

        last_position_id = decoder_input_audio_ids.shape[1] - 1

        valid_text_length = 0
        valid_audio_length = 0

        for i in tqdm.tqdm(
            range(max_new_tokens), desc="Generating tokens", disable=False
        ):
            audio_logits, text_logits, past_key_values = self.alm.forward(
                input_ids=decoder_input_audio_ids,
                text_input_ids=decoder_input_text_ids,
                whisper_input_feature=decoder_input_whisper_feature,
                is_continuous_mask=decoder_is_continuous_mask,
                position_ids=decoder_position_ids,
                past_key_values=past_key_values,
                return_dict=False,
            )

            # Sample text token using the sampler
            next_token_text = sampler.sample_text_logits(
                text_logits, recent_tokens=text_previous_tokens[:i] if i > 0 else None
            )

            # Sample audio token using the sampler
            next_audio_token = sampler.sample_audio_logits(
                audio_logits, recent_tokens=previous_audio_tokens[:i] if i > 0 else None
            )

            if text_stream_is_finished:
                next_token_text.fill_(self.extra_tokens.kimia_text_blank)
            elif next_token_text.item() == self.extra_tokens.kimia_text_eos:
                text_stream_is_finished = True
            else:
                valid_text_length += 1

            text_previous_tokens[i : i + 1] = next_token_text

            if i < self.kimia_text_audiodelaytokens:
                next_audio_token.fill_(self.extra_tokens.kimia_text_blank)
            else:
                if output_type == "text":
                    next_audio_token.fill_(self.extra_tokens.kimia_text_blank)
                else:
                    valid_audio_length += 1

            previous_audio_tokens[i : i + 1] = next_audio_token

            audio_stream_is_finished = next_audio_token.item() in self.eod_ids

            if (
                output_type == "text"
                and text_stream_is_finished
                or output_type == "both"
                and audio_stream_is_finished
            ):
                return_text_tokens = (
                    text_previous_tokens[:valid_text_length]
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )
                return_audio_tokens = (
                    previous_audio_tokens[
                        self.kimia_text_audiodelaytokens : valid_audio_length
                        + self.kimia_text_audiodelaytokens
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )
                return return_audio_tokens, return_text_tokens
            else:
                decoder_input_audio_ids = next_audio_token.unsqueeze(1)
                decoder_input_text_ids = next_token_text.unsqueeze(1)

                decoder_position_ids = (
                    torch.zeros(1, 1, device=torch.cuda.current_device())
                    .fill_(last_position_id + 1)
                    .long()
                    .view(1, 1)
                )
                last_position_id += 1

                decoder_input_whisper_feature = None
                decoder_is_continuous_mask = None

        return_text_tokens = (
            text_previous_tokens[:valid_text_length].detach().cpu().numpy().tolist()
        )
        return_audio_tokens = (
            previous_audio_tokens[
                self.kimia_text_audiodelaytokens : valid_audio_length
                + self.kimia_text_audiodelaytokens
            ]
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )
        return return_audio_tokens, return_text_tokens

    @torch.inference_mode()
    def generate(
        self,
        chats: list[dict],
        output_type="text",
        audio_temperature=0.0,
        audio_top_k=5,
        text_temperature=0.0,
        text_top_k=5,
        audio_repetition_penalty=1.0,
        audio_repetition_window_size=64,
        text_repetition_penalty=1.0,
        text_repetition_window_size=16,
        max_new_tokens=-1,
    ):
        ## TODO: 需要一个check函数，检查输入的history格式是否合法
        ## 比如，对于ASR任务，一定是: text-instruction/audio-instruction + audio-content, 我理解content和instruction是不能换位置的
        ## assistant前必须有user等等，我觉得最好做一下check

        assert output_type in ["text", "both"]

        history = self.prompt_manager.get_prompt(chats, output_type=output_type)

        audio_input_ids, text_input_ids, is_continuous_mask = history.to_tensor()
        audio_features = history.continuous_feature

        generated_wav_tokens = []
        generated_text_tokens = []

        if output_type == "both":
            max_new_tokens = int(12.5 * 120) - audio_input_ids.shape[1]
        else:
            if max_new_tokens == -1:
                max_new_tokens = 7500 - audio_input_ids.shape[1]

        audio_input_ids = audio_input_ids.to(torch.cuda.current_device())
        text_input_ids = text_input_ids.to(torch.cuda.current_device())
        is_continuous_mask = is_continuous_mask.to(torch.cuda.current_device())
        audio_features = [f.to(torch.cuda.current_device()) for f in audio_features]

        generated_wav_tokens, generated_text_tokens = self._generate_loop(
            audio_input_ids=audio_input_ids,
            text_input_ids=text_input_ids,
            max_new_tokens=max_new_tokens,
            audio_temperature=audio_temperature,
            audio_top_k=audio_top_k,
            audio_repetition_penalty=audio_repetition_penalty,
            audio_repetition_window_size=audio_repetition_window_size,
            text_top_k=text_top_k,
            text_temperature=text_temperature,
            text_repetition_penalty=text_repetition_penalty,
            text_repetition_window_size=text_repetition_window_size,
            is_continuous_mask=is_continuous_mask,
            continous_feature=audio_features,
            output_type=output_type,
        )

        generated_wav_tokens = [
            t for t in generated_wav_tokens if t >= self.kimia_token_offset
        ]  #  filter out the illegal tokens

        generated_wav_tokens = torch.tensor(generated_wav_tokens).unsqueeze(0)
        generated_wav_tokens = generated_wav_tokens - self.kimia_token_offset

        generated_text_tokens = [
            t for t in generated_text_tokens if t < self.kimia_token_offset
        ]
        generated_text = self.detokenize_text(generated_text_tokens)
        if self.detokenizer is not None and output_type == "both":
            generated_wav = self.detokenize_audio(generated_wav_tokens)
        else:
            generated_wav = None

        return generated_wav, generated_text

    def detokenize_audio(self, audio_tokens):
        if self.detokenizer is None:
            raise ValueError("Detokenizer is not initialized")
        self.detokenizer.clear_states()
        chunk_size = 30  # hard-coded right now
        first_chunk_size = 30
        cache_speech_collection = []
        audio_tokens = audio_tokens.to(torch.cuda.current_device())
        audio_tokens = audio_tokens.long()
        num_audio_tokens = audio_tokens.size(1)
        first_chunk_semantic_tokens = audio_tokens[:, :first_chunk_size]
        gen_speech = self.detokenizer.detokenize_streaming(
            first_chunk_semantic_tokens,
            is_final=(num_audio_tokens <= first_chunk_size),
            upsample_factor=4,
        )
        cache_speech_collection.append(gen_speech)

        if num_audio_tokens > first_chunk_size:
            res_semantic_tokens = audio_tokens[:, first_chunk_size:]
            for i in range(0, res_semantic_tokens.size(1), chunk_size):
                chunk_semantic_tokens = res_semantic_tokens[:, i : i + chunk_size]
                gen_speech = self.detokenizer.detokenize_streaming(
                    chunk_semantic_tokens,
                    upsample_factor=4,
                    is_final=(i + chunk_size >= res_semantic_tokens.size(1)),
                )
                cache_speech_collection.append(gen_speech)

        gen_speech = torch.cat(cache_speech_collection, dim=-1)
        return gen_speech

    def detokenize_text(self, text_tokens):
        valid_text_ids = []
        for x in text_tokens:
            if x == self.extra_tokens.kimia_text_eos:
                break
            valid_text_ids.append(x)
        return self.prompt_manager.text_tokenizer.decode(valid_text_ids)

    def detokenize_audio_chunk(self, audio_tokens, is_final=False):
        """Detokenize a chunk of audio tokens for streaming"""
        if self.detokenizer is None:
            raise ValueError("Detokenizer is not initialized")
            
        audio_tokens = audio_tokens.to(torch.cuda.current_device())
        audio_tokens = audio_tokens.long()
        
        gen_speech = self.detokenizer.detokenize_streaming(
            audio_tokens,
            is_final=is_final,
            upsample_factor=4,
        )
        return gen_speech
        
    @torch.inference_mode()
    def generate_stream(
        self,
        chats: list[dict],
        output_type="both",
        audio_temperature=0.0,
        audio_top_k=5,
        text_temperature=0.0,
        text_top_k=5,
        audio_repetition_penalty=1.0,
        audio_repetition_window_size=64,
        text_repetition_penalty=1.0,
        text_repetition_window_size=16,
        max_new_tokens=-1,
    ) -> Generator[Tuple[Optional[torch.Tensor], Optional[str]], None, None]:
        """Generate audio and text in a streaming fashion"""
        assert output_type in ["text", "both"]
        assert self.detokenizer is not None or output_type == "text", "Detokenizer must be initialized for audio output"

        # Initialize timers for performance logging
        start_time = time.time()
        first_audio_token_time = None
        first_audio_chunk_time = None
        first_text_token_time = None

        logger.info(f"Starting streaming generation with output_type={output_type}")

        history = self.prompt_manager.get_prompt(chats, output_type=output_type)
        logger.info(f"Prompt preparation took {time.time() - start_time:.2f}s")

        audio_input_ids, text_input_ids, is_continuous_mask = history.to_tensor()
        logger.info(f"Prompt audio input ids shape: {audio_input_ids.shape[1]}, text input ids shape: {text_input_ids.shape[1]}")
        audio_features = history.continuous_feature

        if output_type == "both":
            max_new_tokens = int(12.5 * 120) - audio_input_ids.shape[1]
        else:
            if max_new_tokens == -1:
                max_new_tokens = 7500 - audio_input_ids.shape[1]

        logger.info(f"Will generate up to {max_new_tokens} new tokens")

        # Move tensors to GPU
        audio_input_ids = audio_input_ids.to(torch.cuda.current_device())
        text_input_ids = text_input_ids.to(torch.cuda.current_device())
        is_continuous_mask = is_continuous_mask.to(torch.cuda.current_device())
        audio_features = [f.to(torch.cuda.current_device()) for f in audio_features]

        # Initialize the streaming generation
        sampler = KimiASampler(
            audio_top_k=audio_top_k,
            audio_temperature=audio_temperature,
            audio_repetition_penalty=audio_repetition_penalty,
            audio_repetition_window_size=audio_repetition_window_size,
            text_top_k=text_top_k,
            text_temperature=text_temperature,
            text_repetition_penalty=text_repetition_penalty,
            text_repetition_window_size=text_repetition_window_size,
        )

        # Initialize state variables
        text_stream_is_finished = False
        audio_stream_is_finished = False
        previous_audio_tokens = torch.zeros(
            (4096,), dtype=torch.int, device=torch.cuda.current_device()
        )
        text_previous_tokens = torch.zeros(
            (4096,), dtype=torch.int, device=torch.cuda.current_device()
        )

        decoder_input_audio_ids = audio_input_ids.clone()
        decoder_input_text_ids = text_input_ids.clone()
        decoder_position_ids = (
            torch.arange(0, decoder_input_audio_ids.shape[1], device=torch.cuda.current_device())
            .unsqueeze(0)
            .long()
        )
        decoder_input_whisper_feature = audio_features
        decoder_is_continuous_mask = is_continuous_mask
        past_key_values = None

        last_position_id = decoder_input_audio_ids.shape[1] - 1

        valid_text_length = 0
        valid_audio_length = 0
        
        # Initialize audio streaming state
        accumulated_audio_tokens = []
        current_text = ""
        
        # Initialize detokenizer state if generating audio
        if output_type == "both":
            self.detokenizer.clear_states()

        generation_start_time = time.time()
        logger.info(f"Generation preparation took {generation_start_time - start_time:.2f}s")
        
        # Counters for logging
        total_audio_tokens = 0
        total_text_tokens = 0
        chunk_counter = 1
        last_chunk_time = None

        # Start generation loop
        for i in tqdm.tqdm(range(max_new_tokens), desc="Generating tokens", disable=False):
            token_gen_start = time.time()
            audio_logits, text_logits, past_key_values = self.alm.forward(
                input_ids=decoder_input_audio_ids,
                text_input_ids=decoder_input_text_ids,
                whisper_input_feature=decoder_input_whisper_feature,
                is_continuous_mask=decoder_is_continuous_mask,
                position_ids=decoder_position_ids,
                past_key_values=past_key_values,
                return_dict=False,
            )

            # Sample text token
            next_token_text = sampler.sample_text_logits(
                text_logits, recent_tokens=text_previous_tokens[:i] if i > 0 else None
            )

            # Sample audio token
            next_audio_token = sampler.sample_audio_logits(
                audio_logits, recent_tokens=previous_audio_tokens[:i] if i > 0 else None
            )

            if i == 0:
                logger.info(f"First token generation took {time.time() - token_gen_start:.2f}s")
       
            # Process text token
            if text_stream_is_finished:
                next_token_text.fill_(self.extra_tokens.kimia_text_blank)
            elif next_token_text.item() == self.extra_tokens.kimia_text_eos:
                text_stream_is_finished = True
                logger.info(f"Text generation finished after {i+1} tokens, taking {time.time() - generation_start_time:.2f}s total")
                # Return the final complete text
                valid_text_ids = [
                    t for t in text_previous_tokens[:valid_text_length].detach().cpu().numpy().tolist() 
                    if t < self.kimia_token_offset
                ]
                current_text = self.prompt_manager.text_tokenizer.decode(valid_text_ids)
                yield None, current_text
            else:
                valid_text_length += 1
                total_text_tokens += 1
                
                if first_text_token_time is None:
                    first_text_token_time = time.time()
                    logger.info(f"First text token generated after {first_text_token_time - generation_start_time:.2f}s")
                
                # Update partial text if certain conditions are met (e.g., every 5 tokens or at punctuation)
                if valid_text_length % 5 == 0 or next_token_text.item() in [46, 33, 63, 58]:  # Common punctuation token IDs
                    valid_text_ids = [
                        t for t in text_previous_tokens[:valid_text_length].detach().cpu().numpy().tolist() 
                        if t < self.kimia_token_offset
                    ]
                    current_text = self.prompt_manager.text_tokenizer.decode(valid_text_ids)
                    yield None, current_text

            text_previous_tokens[i : i + 1] = next_token_text

            # Process audio token
            if i < self.kimia_text_audiodelaytokens:
                next_audio_token.fill_(self.extra_tokens.kimia_text_blank)
            else:
                if output_type == "text":
                    next_audio_token.fill_(self.extra_tokens.kimia_text_blank)
                else:
                    valid_audio_length += 1
                    
                    # Track valid audio tokens for streaming
                    if next_audio_token.item() >= self.kimia_token_offset:
                        if first_audio_token_time is None:
                            first_audio_token_time = time.time()
                            logger.info(f"First audio token generated after {first_audio_token_time - generation_start_time:.2f}s,"
                                        f"text tokens count: {valid_text_length}")
                            if first_text_token_time is not None:
                                logger.info(f"Audio started {first_audio_token_time - first_text_token_time:.2f}s after text")
                        
                        accumulated_audio_tokens.append(next_audio_token.item() - self.kimia_token_offset)
                        total_audio_tokens += 1

            previous_audio_tokens[i : i + 1] = next_audio_token
            
            # Check if audio generation is complete
            audio_stream_is_finished = next_audio_token.item() in self.eod_ids
            if audio_stream_is_finished and output_type == "both":
                logger.info(f"Audio generation finished after {i+1} iterations, generating {total_audio_tokens} audio tokens")
                logger.info(f"Audio generation took {time.time() - generation_start_time:.2f}s total")

            # Stream audio when we have enough tokens
            if output_type == "both" and len(accumulated_audio_tokens) >= self.stream_chunk_size:
                audio_chunk_tensor = torch.tensor([accumulated_audio_tokens], device=torch.cuda.current_device())
                
                chunk_decode_start = time.time()
                gen_speech = self.detokenize_audio_chunk(audio_chunk_tensor, is_final=audio_stream_is_finished)
                
                if first_audio_chunk_time is None:
                    first_audio_chunk_time = time.time()
                    logger.info(f"First audio chunk ({len(accumulated_audio_tokens)} tokens) available after {first_audio_chunk_time - generation_start_time:.2f}s")
                    if first_audio_token_time is not None:
                        logger.info(f"Delay between first audio token and first audio chunk: {first_audio_chunk_time - first_audio_token_time:.2f}s")
                    logger.info(f"Audio chunk decoding took {first_audio_chunk_time - chunk_decode_start:.2f}s")
                else:
                    logger.info(f"Audio chunk ({len(accumulated_audio_tokens)} tokens) available after {time.time() - chunk_decode_start:.2f}s")                
                chunk_counter += 1

                if last_chunk_time is not None:
                    logger.info(f"Audio chunk [{chunk_counter}] ,tokens count: {len(accumulated_audio_tokens)} , "
                                f"took  {time.time() - last_chunk_time:.2f}s, decoding took {time.time() - chunk_decode_start:.2f}s")

                last_chunk_time = time.time()
                accumulated_audio_tokens = []  # Reset accumulated tokens
                yield gen_speech, None

            # Check if generation is complete
            if (output_type == "text" and text_stream_is_finished) or (output_type == "both" and audio_stream_is_finished):
                # Return any remaining audio tokens
                if output_type == "both" and accumulated_audio_tokens:
                    audio_chunk_tensor = torch.tensor([accumulated_audio_tokens], device=torch.cuda.current_device())
                    chunk_decode_start = time.time()
                    gen_speech = self.detokenize_audio_chunk(audio_chunk_tensor, is_final=True)
                    logger.info(f"Final audio chunk ({len(accumulated_audio_tokens)} tokens) decoding took {time.time() - chunk_decode_start:.2f}s")
                    yield gen_speech, None
                
                # Final yield with None to signal completion
                logger.info(f"Generation complete: produced {total_text_tokens} text tokens and {total_audio_tokens} audio tokens")
                logger.info(f"Total generation time: {time.time() - start_time:.2f}s")
                yield None, None
                break
            
            # Update decoder inputs for next iteration
            decoder_input_audio_ids = next_audio_token.unsqueeze(1)
            decoder_input_text_ids = next_token_text.unsqueeze(1)
            
            decoder_position_ids = (
                torch.zeros(1, 1, device=torch.cuda.current_device())
                .fill_(last_position_id + 1)
                .long()
                .view(1, 1)
            )
            last_position_id += 1
            
            decoder_input_whisper_feature = None
            decoder_is_continuous_mask = None

        # If we reached max_new_tokens without finishing
        if not text_stream_is_finished and not audio_stream_is_finished:
            logger.info(f"Reached max tokens limit ({max_new_tokens}) without completing generation")
            
            # Return any remaining audio tokens
            if output_type == "both" and accumulated_audio_tokens:
                audio_chunk_tensor = torch.tensor([accumulated_audio_tokens], device=torch.cuda.current_device())
                gen_speech = self.detokenize_audio_chunk(audio_chunk_tensor, is_final=True)
                yield gen_speech, None
            
            # Return final text
            valid_text_ids = [
                t for t in text_previous_tokens[:valid_text_length].detach().cpu().numpy().tolist() 
                if t < self.kimia_token_offset
            ]
            current_text = self.prompt_manager.text_tokenizer.decode(valid_text_ids)
            yield None, current_text
            
            # Final yield with None to signal completion
            logger.info(f"Generation truncated: produced {total_text_tokens} text tokens and {total_audio_tokens} audio tokens")
            logger.info(f"Total generation time: {time.time() - start_time:.2f}s")
            yield None, None
