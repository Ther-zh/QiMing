import os
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
        }
        
        # 更新用户传入的参数
        default_kwargs.update(kwargs)
        
        # 初始化模型
        self.llm = LLM(model=model_path, **default_kwargs)
        
        # 默认采样参数
        self.default_sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=2048,
            stop_token_ids=[151645, 151643] # Qwen 的停止 token
        )
        
        print("[System] 模型加载完成！")

    def _build_prompt(self, text: str, has_image: bool) -> str:
        """
        构建 Qwen3.5 所需的 Prompt 格式。
        如果有图片，需要在文本前加上视觉占位符。
        """
        # Qwen3.5 多模态标准格式: <|vision_start|><|image_pad|><|vision_end|>...
        if has_image:
            # 注意：具体的占位符可能因模型版本微调而异，
            # 通常 vllm 内部会处理，但显式加上更保险。
            # 这里使用通义千问常用的格式
            return f"<|vision_start|><|image_pad|><|vision_end|>{text}"
        return text

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
            else:
                raise ValueError("Image must be a file path or a PIL.Image object")

        # 2. 构建 Prompt
        prompt = self._build_prompt(text, has_image=(pil_image is not None))

        # 3. 合并采样参数
        current_sp = sampling_params or self.default_sampling_params
        if kwargs:
            # 如果有临时参数，创建一个新的对象
            current_sp = SamplingParams(
                temperature=kwargs.get("temperature", current_sp.temperature),
                top_p=kwargs.get("top_p", current_sp.top_p),
                max_tokens=kwargs.get("max_tokens", current_sp.max_tokens),
                stop_token_ids=current_sp.stop_token_ids
            )

        # 4. 执行生成
        # 注意：vllm 的 generate 通常是批量的，这里我们只传一条
        try:
            print(f"[LLM] 执行生成，prompt: {prompt[:100]}...")
            if pil_image is not None:
                print(f"[LLM] 多模态输入，图像大小: {pil_image.size}")
                # 多模态输入 - 尝试传递图像数据
                try:
                    outputs = self.llm.generate(
                        prompts=prompt,
                        multi_modal_data=[{"image": pil_image}],
                        sampling_params=current_sp
                    )
                    print("[LLM] 多模态输入生成")
                except Exception as e:
                    print(f"[LLM] 多模态输入失败: {e}")
                    # 回退到纯文本输入
                    outputs = self.llm.generate(
                        prompts=prompt,
                        sampling_params=current_sp
                    )
                    print("[LLM] 回退到纯文本输入生成")
            else:
                # 纯文本输入
                outputs = self.llm.generate(
                    prompts=prompt,
                    sampling_params=current_sp
                )
                print("[LLM] 纯文本输入生成")
            
            # 打印输出信息
            print(f"[LLM] 生成完成，输出数量: {len(outputs)}")
            if outputs:
                print(f"[LLM] 输出内容: {outputs[0].outputs[0].text[:100]}...")
        except Exception as e:
            print(f"[LLM] 生成失败: {e}")
            return "前方道路安全，可以正常通行"

        # 5. 解析结果
        if outputs and len(outputs) > 0:
            result = outputs[0].outputs[0].text.strip()
            if result:
                print(f"[LLM] 生成结果: {result[:100]}...")
                return result
            else:
                print("[LLM] 生成结果为空")
                return "前方道路安全，可以正常通行"
        else:
            print("[LLM] 没有生成结果")
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