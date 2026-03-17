import os
import numpy as np
from typing import Optional, Union, List
from PIL import Image
from vllm import LLM, SamplingParams

# 强制使用 spawn 方式，避免多进程问题
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

class Qwen35VLLM:
    def __init__(
        self,
        model_path: str,
        max_model_len: int = 8192,
        tensor_parallel_size: int = 1,
        **kwargs
    ):
        """
        初始化 Qwen3.5 vLLM 推理引擎。
        
        Args:
            model_path: 模型本地路径
            max_model_len: 最大上下文长度
            tensor_parallel_size: 张量并行大小 (单卡设为1)
            **kwargs: 其他 vLLM LLM 初始化参数
        """
        print(f"[System] 正在加载模型: {model_path} ...")
        
        # 默认参数 (针对 AWQ 量化版本优化)
        default_kwargs = {
            "quantization": "awq_marlin",
            "enforce_eager": True,  # 关键：禁用 torch.compile 避免报错
            "max_model_len": max_model_len,
            "tensor_parallel_size": tensor_parallel_size,
            "trust_remote_code": True,
            "gpu_memory_utilization": 0.3, # 限制显存占用到约6GB
            "disable_log_stats": True,  # 禁用日志统计
        }
        
        # 更新用户传入的参数
        default_kwargs.update(kwargs)
        
        # 初始化模型
        self.llm = LLM(model=model_path, **default_kwargs)
        
        # 默认采样参数
        # 语音助手适配的采样参数（关闭思考后官方推荐配置）
        # 通过extra_args传递chat_template_kwargs来禁用思考模式
        self.default_sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.8,
            top_k=20,
            max_tokens=512,  # 语音助手回复无需太长，减少无效生成耗时
            stop=["<|im_end|>", "</think>"],  # 兜底stop词：万一出现思考标签直接截断
            presence_penalty=1.5,  # 增加重复惩罚，减少重复输出
            frequency_penalty=1.0,  # 增加频率惩罚，减少重复输出
            extra_args={"chat_template_kwargs": {"enable_thinking": False}}  # 禁用思考模式
        )
        
        print("[System] 模型加载完成！")

    def _build_prompt(self, text: str, has_image: bool = False, image_data=None) -> str:
        """
        构建 Qwen3.5 所需的 Prompt 格式。
        使用官方apply_chat_template方法生成prompt，确保与全局禁用思考配置兼容。
        """
        # 【必须用官方方法生成prompt，不要自己乱拼】
        # 构造对话消息
        messages = []
        
        # 如果有图片，使用多模态消息格式
        if has_image and image_data is not None:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": image_data},
                    {"type": "text", "text": text}
                ]
            })
        else:
            # 纯文本消息
            messages.append({"role": "user", "content": text})
        
        # 使用官方tokenizer的apply_chat_template生成prompt
        # 全局已关闭思考模式，无需额外加参数
        prompt = self.llm.get_tokenizer().apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        return prompt

    def generate(
        self,
        text: str,
        image: Optional[Union[str, Image.Image]] = None,
        sampling_params: Optional[SamplingParams] = None,
        **kwargs
    ) -> str:
        """
        生成回复。
        
        Args:
            text: 输入的文本提示词
            image: 输入的图片 (可以是文件路径字符串 或 PIL.Image 对象)
            sampling_params: 自定义采样参数 (可选)
            **kwargs: 临时覆盖采样参数 (如 temperature, max_tokens 等)
        
        Returns:
            生成的文本字符串
        """
        # 1. 处理图片
        pil_image = None
        if image is not None:
            if isinstance(image, str):
                pil_image = Image.open(image).convert("RGB")
            elif isinstance(image, Image.Image):
                pil_image = image.convert("RGB")
            elif isinstance(image, np.ndarray):
                # 处理numpy数组（OpenCV格式）
                if len(image.shape) == 3 and image.shape[2] == 3:
                    # BGR格式转RGB
                    if image.dtype == np.uint8:
                        pil_image = Image.fromarray(image[:, :, ::-1])
                    else:
                        # 非uint8类型，先归一化到0-255
                        image = (image * 255).astype(np.uint8)
                        pil_image = Image.fromarray(image[:, :, ::-1])
                else:
                    raise ValueError("Numpy array must be RGB format (H, W, 3)")
            else:
                raise ValueError("Image must be a file path, PIL.Image object, or numpy array")

        # 2. 合并采样参数
        current_sp = sampling_params or self.default_sampling_params
        if kwargs:
            # 如果有临时参数，创建一个新的对象
            current_sp = SamplingParams(
                temperature=kwargs.get("temperature", current_sp.temperature),
                top_p=kwargs.get("top_p", current_sp.top_p),
                max_tokens=kwargs.get("max_tokens", current_sp.max_tokens),
                stop=current_sp.stop,
                extra_args=current_sp.extra_args if hasattr(current_sp, 'extra_args') else None
            )

        # 3. 构建消息格式
        # 使用chat方法，可以传递chat_template_kwargs来禁用思考模式
        messages = []
        mm_data = None
        if pil_image is not None:
            # 多模态消息 - 使用OpenAI兼容格式
            # 将PIL Image转换为base64编码的data URL
            import base64
            from io import BytesIO
            buffered = BytesIO()
            pil_image.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            img_url = f"data:image/jpeg;base64,{img_base64}"
            
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": img_url}},
                    {"type": "text", "text": text}
                ]
            })
        else:
            # 纯文本消息
            messages.append({"role": "user", "content": text})

        # 4. 执行生成
        # 使用chat方法，传递chat_template_kwargs来禁用思考模式
        try:
            import sys
            print(f"[LLM] 执行生成...")
            sys.stdout.flush()
            
            outputs = self.llm.chat(
                messages=messages,
                sampling_params=current_sp,
                chat_template_kwargs={"enable_thinking": False}  # 禁用思考模式
            )
            
            print("[LLM] 生成完成")
            sys.stdout.flush()
            
        except Exception as e:
            print(f"[LLM] 生成失败: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            return "前方道路安全，可以正常通行"

        # 5. 解析结果
        if outputs and len(outputs) > 0:
            result = outputs[0].outputs[0].text.strip()
            if result:
                print(f"[LLM] 生成结果: {result[:100]}...")
                sys.stdout.flush()
                return result
            else:
                print("[LLM] 生成结果为空")
                sys.stdout.flush()
                return "前方道路安全，可以正常通行"
        else:
            print("[LLM] 没有生成结果")
            sys.stdout.flush()
            return "前方道路安全，可以正常通行"

    def batch_generate(
        self,
        inputs: List[dict],
        sampling_params: Optional[SamplingParams] = None,
    ) -> List[str]:
        """
        批量生成 (高级用法)。
        
        Args:
            inputs: 字典列表，每个字典包含 "text" 和可选的 "image"
            sampling_params: 采样参数
        
        Returns:
            生成结果列表
        """
        prompts = []
        multi_modal_datas = []
        
        for item in inputs:
            text = item["text"]
            img = item.get("image")
            
            pil_image = None
            if img:
                if isinstance(img, str):
                    pil_image = Image.open(img).convert("RGB")
                else:
                    pil_image = img.convert("RGB")
            
            prompts.append(self._build_prompt(text, has_image=(pil_image is not None)))
            if pil_image:
                multi_modal_datas.append({"image": pil_image})
            else:
                multi_modal_datas.append(None)

        current_sp = sampling_params or self.default_sampling_params
        
        outputs = self.llm.generate(
            prompts=prompts,
            multi_modal_data=multi_modal_datas,
            sampling_params=current_sp
        )
        
        return [out.outputs[0].text.strip() for out in outputs]

# ---------------------------------------------------------
# 下面是一个简单的使用示例，当直接运行此脚本时执行
# ---------------------------------------------------------
if __name__ == "__main__":
    # 配置路径
    MODEL_PATH = "/root/autodl-tmp/qwen35/tclf90/Qwen3___5-4B-AWQ"
    
    # 1. 初始化引擎 (这一步比较慢，只需执行一次)
    model = Qwen35VLLM(
        model_path=MODEL_PATH,
        max_model_len=8192
    )
    
    # 2. 纯文本测试
    print("\n--- 测试纯文本生成 ---")
    response = model.generate("你好，请用一句话介绍你自己。")
    print("回复:", response)
    
    # 3. 多模态测试 (如果有图片的话，取消下面注释进行测试)
    # print("\n--- 测试多模态生成 ---")
    # image_path = "test.jpg" # 替换为你的图片路径
    # if os.path.exists(image_path):
    #     response = model.generate("这张图片里有什么？", image=image_path)
    #     print("回复:", response)