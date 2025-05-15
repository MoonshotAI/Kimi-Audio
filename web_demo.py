import os
import json
import gradio as gr
import torch
import numpy as np
import soundfile as sf
import argparse
import logging
import time
import base64
import tempfile
from datetime import datetime
from kimia_infer.api.kimia import KimiAudio
from contextlib import contextmanager

def setup_logging(log_level=logging.INFO, log_file=None):
    """设置日志配置"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 配置根日志记录器
    if log_file:
        logging.basicConfig(level=log_level, format=log_format,
                            handlers=[
                                logging.FileHandler(log_file),
                                logging.StreamHandler()
                            ])
    else:
        logging.basicConfig(level=log_level, format=log_format)
    
    # 返回日志记录器
    return logging.getLogger("kimi-audio-web")

def parse_args():
    parser = argparse.ArgumentParser(description="Kimi-Audio Web Demo")
    parser.add_argument("--model_path", type=str, default="moonshotai/Kimi-Audio-7B-Instruct", 
                        help="模型路径")
    parser.add_argument("--output_dir", type=str, default="test_audios/output", 
                        help="输出文件保存目录")
    parser.add_argument("--port", type=int, default=7860, 
                        help="运行Gradio应用的端口")
    parser.add_argument("--share", action="store_true", 
                        help="是否共享Gradio应用")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="设置日志级别")
    parser.add_argument("--log_file", type=str, default=None,
                        help="日志文件路径。如果不设置，只记录到控制台")
    parser.add_argument("--stream", action="store_true", 
                        help="是否使用流式生成模式")
    parser.add_argument("--first_chunk_size", type=int, default=30, 
                        help="流式模式下，首个音频块的token数量，较大的值可以减少初始延迟感")
    parser.add_argument("--stream_chunk_size", type=int, default=20, 
                        help="流式模式下，首个音频块后每个音频块的token数量")
    return parser.parse_args()

class KimiAudioChat:
    """处理Kimi Audio聊天会话的类"""
    
    # 默认采样参数
    DEFAULT_SAMPLING_PARAMS = {
        "audio_temperature": 0.8,
        "audio_top_k": 10,
        "text_temperature": 0.0,
        "text_top_k": 5,
        "audio_repetition_penalty": 1.0,
        "audio_repetition_window_size": 64,
        "text_repetition_penalty": 1.0,
        "text_repetition_window_size": 16,
    }
    
    # 音频采样率
    AUDIO_SAMPLE_RATE = 22050
    
    def __init__(self, model, output_dir, logger, use_stream=False, first_chunk_size=30, stream_chunk_size=20):
        """初始化聊天会话处理器
        
        Args:
            model: Kimi Audio模型实例
            output_dir: 输出文件保存目录
            logger: 日志记录器
            use_stream: 是否使用流式生成
            first_chunk_size: 首个音频块的token数量
            stream_chunk_size: 后续音频块的token数量
        """
        self.model = model
        self.output_dir = output_dir
        self.logger = logger
        self.use_stream = use_stream
        self.first_chunk_size = first_chunk_size
        self.stream_chunk_size = stream_chunk_size
        self.sampling_params = self.DEFAULT_SAMPLING_PARAMS.copy()
        self.output_type = "both"
        
        # 如果使用流式模式，设置流式chunk大小
        if use_stream and hasattr(model, 'stream_chunk_size'):
            # 初始时使用first_chunk_size
            model.stream_chunk_size = first_chunk_size
            logger.info(f"设置流式模式 first_chunk_size 为 {first_chunk_size}, 后续 chunk_size 为 {stream_chunk_size}")
    
    def update_params(self, params):
        """更新采样参数"""
        self.sampling_params.update(params)
    
    def set_output_type(self, output_type):
        """设置输出类型"""
        self.output_type = output_type
    
    def set_stream_mode(self, use_stream):
        """设置是否使用流式模式
        
        Args:
            use_stream: 是否启用流式生成
        """
        self.use_stream = use_stream
        
        # 如果启用流式模式，配置模型的流式参数
        if use_stream and hasattr(self.model, 'stream_chunk_size'):
            # 启用流式模式时，设置为first_chunk_size
            self.model.stream_chunk_size = self.first_chunk_size
            self.logger.info(f"已配置流式生成参数: first_chunk_size={self.first_chunk_size}, stream_chunk_size={self.stream_chunk_size}")
        elif not use_stream:
            self.logger.info("已设置为非流式生成模式")
    
    @contextmanager
    def _request_context(self, label):
        """请求上下文管理器，用于计时和日志记录"""
        request_id = f"req_{int(time.time())}_{os.getpid()}"
        self.logger.info(f"[{request_id}] 开始 {label}")
        start_time = time.time()
        try:
            yield request_id
        finally:
            self.logger.info(f"[{request_id}] 完成 {label}，耗时 {time.time() - start_time:.2f}秒")
    
    def _build_messages_from_history(self, history, request_id):
        """从聊天历史构建消息列表"""
        messages = []
        
        if history:
            self.logger.info(f"[{request_id}] 处理历史记录: {len(history)} 条消息")
            
        for h_item in history:
            if h_item["role"] == "user":
                if isinstance(h_item["content"], tuple):
                    messages.append({"role": "user", "message_type": "audio", "content": h_item["content"][0]})
                elif h_item["content"]:
                    messages.append({"role": "user", "message_type": "text", "content": h_item["content"]})
            else:
                if isinstance(h_item["content"], str):
                    messages.append({"role": "assistant", "message_type": "text", "content": h_item["content"]})
                
        return messages
    
    def _create_empty_audio_chunk(self):
        """创建空音频块，用于流式生成初始化"""
        # 确保返回的是长度为1的数组，避免后续处理时的尺寸不匹配问题
        return (self.AUDIO_SAMPLE_RATE, np.zeros(1, dtype=np.int16))
    
    def chat(self, message, history):
        """ChatInterface格式的聊天函数，处理标准的非流式生成
        
        Args:
            message (dict): 用户输入的文本消息
            history (list): 聊天历史记录
            
        Returns:
            response(List): 生成的文本回复和音频数据
        """
        with self._request_context("处理标准聊天请求") as request_id:
            input_text = message["text"]
            input_audio_path = None
            
            if message["files"]:
                input_audio_path = message["files"][0]
                
            self.logger.info(f"[{request_id}] 输入文本: {input_text}, 输入音频: {input_audio_path}")
            
            # 构建消息列表
            messages = self._build_messages_from_history(history, request_id)
            
            # 处理当前用户输入
            user_has_input = False
            
            # 添加文本消息
            if input_text:
                messages.append({"role": "user", "message_type": "text", "content": input_text})
                user_has_input = True
            
            # 处理音频输入        
            if input_audio_path is not None:
                input_audio_path = self._process_input_audio(input_audio_path, request_id)
                if input_audio_path:
                    messages.append({"role": "user", "message_type": "audio", "content": input_audio_path})
                    user_has_input = True
            
            # 如果用户没有提供任何输入
            if not user_has_input:
                return ["请提供文本或语音消息"]
            
            # 记录发送给模型的消息
            for i, msg in enumerate(messages):
                self.logger.info(f"[{request_id}] 消息 {i}: {msg['role']}, {msg['message_type']}, "
                                f"{msg['content'][:30]}..." if len(msg['content']) > 30 else msg['content'])
            
            try:
                # 使用非流式生成方法
                return self._generate_normal(messages, request_id)
                    
            except Exception as e:
                self.logger.error(f"[{request_id}] 处理请求时出错: {str(e)}", exc_info=True)
                return [f"处理错误: {str(e)}"]
    
    def chat_stream(self, message, history):
        """ChatInterface格式的聊天函数，处理流式生成
        
        Args:
            message (dict): 用户输入的文本消息
            history (list): 聊天历史记录
            
        Yields:
            response(List), audio_chunk: 生成的文本回复和音频数据
        """
        with self._request_context("处理流式聊天请求") as request_id:
            input_text = message["text"]
            input_audio_path = None
            
            if message["files"]:
                input_audio_path = message["files"][0]
                
            self.logger.info(f"[{request_id}] 输入文本: {input_text}, 输入音频: {input_audio_path}")
            
            # 构建消息列表
            messages = self._build_messages_from_history(history, request_id)
            
            # 处理当前用户输入
            user_has_input = False
            
            # 添加文本消息
            if input_text:
                messages.append({"role": "user", "message_type": "text", "content": input_text})
                user_has_input = True
            
            # 处理音频输入        
            if input_audio_path is not None:
                input_audio_path = self._process_input_audio(input_audio_path, request_id)
                if input_audio_path:
                    messages.append({"role": "user", "message_type": "audio", "content": input_audio_path})
                    user_has_input = True
            
            # 如果用户没有提供任何输入
            if not user_has_input:
                empty_chunk = self._create_empty_audio_chunk()
                yield ["请提供文本或语音消息"], empty_chunk
                return
            
            # 记录发送给模型的消息
            for i, msg in enumerate(messages):
                self.logger.info(f"[{request_id}] 消息 {i}: {msg['role']}, {msg['message_type']}, "
                                f"{msg['content'][:30]}..." if len(msg['content']) > 30 else msg['content'])
            
            try:
                # 使用流式生成方法
                yield from self._generate_stream_ui(messages, request_id)
                    
            except Exception as e:
                self.logger.error(f"[{request_id}] 处理请求时出错: {str(e)}", exc_info=True)
                empty_chunk = self._create_empty_audio_chunk()
                yield [f"处理错误: {str(e)}"], empty_chunk
                
    def _generate_normal(self, messages, request_id):
        """使用非流式模式生成回复"""
        self.logger.info(f"[{request_id}] 使用标准模式生成回复...")
        start_time = time.time()
        
        wav, text = self.model.generate(
            messages, 
            **self.sampling_params, 
            output_type=self.output_type
        )
        
        generation_time = time.time() - start_time
        self.logger.info(f"[{request_id}] 生成完成，耗时 {generation_time:.2f} 秒")
        
        # 处理响应
        bot_response = text if text else "未生成文本回复"
        self.logger.info(f"[{request_id}] 生成的文本: {bot_response}")
        response = [bot_response]
        
        # 处理音频输出（如果有）
        if self.output_type == "both" and wav is not None:
            output_path = os.path.join(self.output_dir, f"{request_id}_output.wav")
            self.logger.info(f"[{request_id}] 保存输出音频到 {output_path}")
            
            audio_data = wav.detach().cpu().view(-1).numpy()
            sf.write(output_path, audio_data, self.AUDIO_SAMPLE_RATE)
            response.append(gr.Audio(output_path, label="AI语音回复", autoplay=True, interactive=False, format="wav"))
        
        return response
    
    def _generate_stream_ui(self, messages, request_id):
        """使用流式模式生成回复，并实时更新UI"""
        self.logger.info(f"[{request_id}] 使用流式模式生成回复...")
        start_time = time.time()
        
        # 准备临时目录用于存放音频块
        with tempfile.TemporaryDirectory(dir=self.output_dir, prefix=f"stream_{request_id}_") as temp_dir:
            # 初始化变量
            current_text = ""
            audio_chunks_paths = []
            chunk_counter = 0
            latest_audio_path = None
            first_audio_time = None
            empty_chunk = self._create_empty_audio_chunk()
            
            # 首先，只返回一个加载指示
            yield ["正在生成回复..."], empty_chunk
            self.model.stream_chunk_size = self.first_chunk_size
            # 开始流式生成
            for audio_chunk, text_chunk in self.model.generate_stream(
                messages, 
                **self.sampling_params, 
                output_type=self.output_type
            ):
                # 处理音频块
                if audio_chunk is not None:
                    chunk_counter += 1
                    
                    # 首个音频块生成后，切换到后续的chunk size
                    if chunk_counter == 1 and hasattr(self.model, 'stream_chunk_size'):
                        self.model.stream_chunk_size = self.stream_chunk_size
                        self.logger.info(f"[{request_id}] 首个音频块生成后，调整为常规chunk size: {self.stream_chunk_size}")
                    
                    if first_audio_time is None:
                        first_audio_time = time.time()
                        self.logger.info(f"[{request_id}] 首个音频块生成延迟: {first_audio_time - start_time:.2f}秒")
                    
                    # 保存当前音频块
                    chunk_path = os.path.join(temp_dir, f"chunk_{chunk_counter}.wav")
                    audio_data = audio_chunk.detach().cpu().view(-1).numpy()
                    sf.write(chunk_path, audio_data, self.AUDIO_SAMPLE_RATE)
                    audio_chunks_paths.append(chunk_path)
                    
                    # 使用最新的音频块作为当前播放内容
                    latest_audio_path = chunk_path
                    duration =len(audio_data) / self.AUDIO_SAMPLE_RATE
                    self.logger.info(f"[{request_id}] 生成音频块 #{chunk_counter}, 时长:{duration:.2f}s 保存到 {chunk_path}")
                    
                    # 更新UI - 发送音频和当前文本
                    response = [current_text]
                    yield response, (self.AUDIO_SAMPLE_RATE, audio_data)
                
                # 处理文本块
                elif text_chunk is not None and text_chunk:
                    current_text = text_chunk
                    self.logger.info(f"[{request_id}] 更新文本: {current_text}")
                    # 只更新文本
                    yield [current_text], empty_chunk
                
                # 结束标志
                if audio_chunk is None and text_chunk is None:
                    self.logger.info(f"[{request_id}] 流式生成完成")
                    break
            
            # 合成最终的完整音频文件（如果有音频块）
            if audio_chunks_paths:
                final_audio_path = os.path.join(self.output_dir, f"{request_id}_final_output.wav")
                
                # 读取并合并所有音频块
                try:
                    audio_segments = []
                    for chunk_path in audio_chunks_paths:
                        data, rate = sf.read(chunk_path)
                        audio_segments.append(data)
                    
                    combined_audio = np.concatenate(audio_segments)
                    sf.write(final_audio_path, combined_audio, self.AUDIO_SAMPLE_RATE)
                    
                    # 最终响应包含完整的文本和完整的音频
                    final_response = [current_text if current_text else "未生成文本回复"]
                    final_response.append(gr.Audio(final_audio_path, label="AI完整语音回复", 
                                                autoplay=False, interactive=False, format="wav"))
                    
                    generation_time = time.time() - start_time
                    self.logger.info(f"[{request_id}] 流式生成完成，总耗时 {generation_time:.2f}秒，保存到 {final_audio_path}")
                    
                    yield final_response, empty_chunk
                except Exception as e:
                    self.logger.error(f"[{request_id}] 合并音频文件失败: {str(e)}", exc_info=True)
                    yield [current_text if current_text else "未生成文本回复"], empty_chunk
            else:
                # 只有文本没有音频的情况
                yield [current_text if current_text else "未生成文本回复"], empty_chunk
    
    def _process_input_audio(self, audio_file, request_id):
        """处理输入音频文件并返回路径"""
        try:
            if isinstance(audio_file, tuple) and len(audio_file) == 2:
                # 录音的音频
                temp_file = os.path.join(self.output_dir, f"{request_id}_input.wav")
                self.logger.info(f"[{request_id}] 保存麦克风输入到 {temp_file}")
                sf.write(temp_file, audio_file[1], audio_file[0])
                return temp_file
            else:
                # 上传的音频文件
                self.logger.info(f"[{request_id}] 使用上传的音频文件: {audio_file}")
                return audio_file
        except Exception as e:
            self.logger.error(f"处理音频输入错误: {str(e)}", exc_info=True)
            return None

def main():
    args = parse_args()
    
    # 设置日志
    log_level = getattr(logging, args.log_level)
    log_file = args.log_file or f"kimi_web_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logging(log_level=log_level, log_file=log_file)
    
    logger.info(f"启动 Kimi-Audio Web Demo")
    logger.info(f"参数: {args}")
    
    # 记录流式模式状态
    if args.stream:
        logger.info(f"已启用流式生成模式, first_chunk_size={args.first_chunk_size}, stream_chunk_size={args.stream_chunk_size}")
    else:
        logger.info(f"使用标准生成模式")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"输出目录: {args.output_dir}")
    
    # 初始化模型
    logger.info(f"从 {args.model_path} 加载模型...")
    start_time = time.time()
    try:
        model = KimiAudio(
            model_path=args.model_path,
            load_detokenizer=True,
        )
        # 如果使用流式模式，设置流式chunk大小
        if args.stream:
            # 初始时使用first_chunk_size
            model.stream_chunk_size = args.first_chunk_size
            
        logger.info(f"模型加载成功，耗时 {time.time() - start_time:.2f} 秒")
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        raise
    
    # 初始化聊天处理器
    chat_handler = KimiAudioChat(
        model, 
        args.output_dir, 
        logger,
        use_stream=args.stream,
        first_chunk_size=args.first_chunk_size,
        stream_chunk_size=args.stream_chunk_size
    )
    
    # 添加CSS自定义样式
    custom_css = """
    audio:focus { outline: none; }
    audio::-webkit-media-controls-panel { background-color: #f1f3f4; }
    .audio-container { transition: all 0.3s ease; }
    .audio-container:hover { background-color: #eef2f5 !important; }
    .message-audio { margin-top: 10px; }
    .chatbot .message.bot .message-audio { display: block; margin-top: 8px; }
    /* 音频播放器样式 */
    .chatbot audio { border-radius: 4px; max-width: 100%; margin-top: 10px; }
    /* 对话气泡内的音频组件 */
    .chatbot .bot audio { background-color: rgba(255, 255, 255, 0.2); }
    """
    
    # 创建Gradio界面
    demo = gr.Blocks(css=custom_css, title="Kimi-Audio 聊天机器人")
    
    with demo:
        gr.Markdown("# Kimi-Audio 语音聊天助手")
        # 仅在流式模式下添加音频输出组件
        if args.stream:
            audio_output = gr.Audio(
                interactive=False, 
                streaming=True, 
                autoplay=True, 
                label="AI语音回复"
            )
        
        with gr.Tab("聊天"):
            # 创建Chat组件，使用Chat Interface支持多模态输出
            # 根据是否启用流式生成选择不同的处理函数
            chat_fn = chat_handler.chat_stream if args.stream else chat_handler.chat
            
            gr.ChatInterface(
                fn=chat_fn,
                type="messages",
                multimodal=True,
                save_history=False,
                textbox=gr.MultimodalTextbox(file_count="multiple", file_types=["audio"], sources=["upload", "microphone"]),   
                chatbot=gr.Chatbot(height=500),
                title="",
                additional_outputs=[audio_output] if args.stream else [],
                description="发送文本或语音消息，AI将回复文本和音频" + (" (流式生成模式已启用)" if args.stream else "")
            )
        
        with gr.Tab("设置"):
            with gr.Accordion("模型参数", open=True):
                with gr.Row():
                    with gr.Column():
                        audio_temperature = gr.Slider(0.0, 2.0, value=0.8, label="音频温度", info="控制音频生成的随机性")
                        audio_top_k = gr.Slider(1, 50, value=10, step=1, label="音频Top K", info="控制每一步考虑的候选音频标记数量")
                        audio_repetition_penalty = gr.Slider(0.1, 5.0, value=1.0, label="音频重复惩罚", info="防止重复音频片段")
                        audio_repetition_window_size = gr.Slider(1, 200, value=64, step=1, label="音频重复窗口大小")
                    
                    with gr.Column():
                        text_temperature = gr.Slider(0.0, 2.0, value=0.0, label="文本温度", info="控制文本生成的随机性")
                        text_top_k = gr.Slider(1, 50, value=5, step=1, label="文本Top K", info="控制每一步考虑的候选文本标记数量")
                        text_repetition_penalty = gr.Slider(0.1, 5.0, value=1.0, label="文本重复惩罚", info="防止重复文本")
                        text_repetition_window_size = gr.Slider(1, 200, value=16, step=1, label="文本重复窗口大小")
                
                output_type = gr.Radio(
                    ["text", "both"], 
                    value="both", 
                    label="输出类型", 
                    info="选择AI应该回复的内容类型：'text'=仅文本, 'both'=文本和音频"
                )
                
                # 仅在流式模式启用时显示流式参数设置
                if args.stream:
                    gr.Markdown("### 流式生成参数")
                    with gr.Row():
                        first_chunk_size = gr.Slider(
                            10, 100, value=args.first_chunk_size, step=5, 
                            label="首个音频块大小", 
                            info="首个音频块的token数量，较大的值可以减少初始等待感，但可能增加首次响应延迟"
                        )
                        
                        later_chunk_size = gr.Slider(
                            5, 50, value=args.stream_chunk_size, step=5, 
                            label="后续音频块大小", 
                            info="后续音频块的token数量，较小的值可以获得更流畅的体验"
                        )
                
                # 更新参数按钮
                update_btn = gr.Button("应用设置", variant="primary")
                result = gr.Textbox(label="状态")
                
                # 功能：当点击更新按钮时，更新聊天处理器的参数
                def update_params(audio_temp, audio_topk, audio_rep_penalty, audio_rep_window,
                                text_temp, text_topk, text_rep_penalty, text_rep_window,
                                out_type, *stream_params):
                    """更新所有模型参数和设置"""
                    # 构建采样参数
                    params = {
                        "audio_temperature": float(audio_temp),
                        "audio_top_k": int(audio_topk),
                        "text_temperature": float(text_temp),
                        "text_top_k": int(text_topk),
                        "audio_repetition_penalty": float(audio_rep_penalty),
                        "audio_repetition_window_size": int(audio_rep_window),
                        "text_repetition_penalty": float(text_rep_penalty),
                        "text_repetition_window_size": int(text_rep_window),
                    }
                    
                    # 提取流式参数（如果有）
                    first_size = stream_params[0] if stream_params else None
                    later_size = stream_params[1] if len(stream_params) > 1 else None
                    
                    # 打印参数以便调试
                    logger.info(f"更新参数: {params}")
                    logger.info(f"输出类型: {out_type}")
                    if first_size is not None and later_size is not None:
                        logger.info(f"流式参数: first_chunk_size={first_size}, stream_chunk_size={later_size}")
                    
                    # 更新采样参数
                    chat_handler.update_params(params)
                    
                    # 更新输出类型
                    chat_handler.set_output_type(out_type)
                    
                    # 更新流式参数（如果是流式模式）
                    if args.stream and first_size is not None and later_size is not None:
                        f_chunk_size = int(first_size)
                        l_chunk_size = int(later_size)
                        
                        chat_handler.first_chunk_size = f_chunk_size
                        chat_handler.stream_chunk_size = l_chunk_size
                        
                        # 设置初始值为first_chunk_size
                        if hasattr(chat_handler.model, 'stream_chunk_size'):
                            chat_handler.model.stream_chunk_size = f_chunk_size
                            logger.info(f"已更新流式参数: first_chunk_size={f_chunk_size}, stream_chunk_size={l_chunk_size}")
                            stream_status = f" (流式参数: 首块大小={f_chunk_size}，后续块大小={l_chunk_size})"
                            return f"✅ 设置已应用{stream_status}"
                    
                    return "✅ 设置已应用"
                
                # 根据当前模式准备输入参数
                if args.stream:
                    # 流式模式下，包含流式参数
                    inputs = [
                        audio_temperature, audio_top_k, audio_repetition_penalty, audio_repetition_window_size,
                        text_temperature, text_top_k, text_repetition_penalty, text_repetition_window_size,
                        output_type, first_chunk_size, later_chunk_size
                    ]
                else:
                    # 非流式模式下，不包含流式参数
                    inputs = [
                        audio_temperature, audio_top_k, audio_repetition_penalty, audio_repetition_window_size,
                        text_temperature, text_top_k, text_repetition_penalty, text_repetition_window_size,
                        output_type
                    ]
                
                # 连接按钮点击事件和更新函数
                update_btn.click(
                    fn=update_params,
                    inputs=inputs,
                    outputs=result
                )
        
        with gr.Tab("关于"):
            gr.Markdown(f"""
            # Kimi-Audio 多模态聊天机器人
            
            这是一个基于 {args.model_path} 模型的多模态聊天机器人，支持语音和文本交互。
            
            ## 功能特点
            
            - **多模态输入**: 可以输入文本或录制音频
            - **多模态输出**: 可以同时输出文本和音频回复
            - **参数调整**: 在设置标签页中调整模型参数
            - **生成模式**: {"流式生成" if args.stream else "标准生成"}
            
            ## 使用说明
            
            1. 在文本框中输入消息，或使用录音按钮录制语音
            2. 点击发送按钮或按回车键发送消息
            3. 机器人将生成文本和语音回复（如果在设置中启用）
            4. 点击音频播放按钮收听回复
            
            ## 模型信息
            
            - 模型: {args.model_path}
            
            项目地址: [GitHub](https://github.com/moonshotai/Kimi-Audio)
            """)
    
    # 启动应用
    logger.info(f"启动Gradio应用，端口: {args.port}, 共享: {args.share}")
    demo.launch(
        server_port=args.port, 
        server_name="0.0.0.0", 
        share=args.share,
        allowed_paths=["*"],
        show_api=False
    )
    logger.info("Gradio应用已关闭")

if __name__ == "__main__":
    main() 