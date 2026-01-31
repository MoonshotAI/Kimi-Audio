import argparse
import os
import json

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from huggingface_hub import snapshot_download
import tqdm

from kimia_infer.api.prompt_manager import KimiAPromptManager


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="moonshotai/Kimi-Audio-7B")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    if os.path.exists(args.model_name_or_path):
        # local path
        cache_path = args.model_name_or_path
    else:
        # cache everything if model_path is a model-id
        cache_path = snapshot_download(args.model_name_or_path)

    # load model config
    model_config = AutoConfig.from_pretrained(cache_path, trust_remote_code=True)

    prompt_manager = KimiAPromptManager(
            model_path=cache_path, kimia_token_offset=model_config.kimia_token_offset, kimia_text_audiodelaytokens=model_config.kimia_mimo_audiodelaytokens
        )
    
    with open(args.input_file, "r") as f, open(args.output_file, "w") as f_out:
        lines = f.readlines()
        for line in tqdm.tqdm(lines):
            data = json.loads(line)
            
            for msg in data["conversation"]:
                if msg["message_type"] == "audio":
                    audio_path = msg["content"]
                    audio_tokens = prompt_manager._tokenize_audio(audio_path)
                    msg["audio_tokens"] = audio_tokens

            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
