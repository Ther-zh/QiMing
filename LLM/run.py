from qwen_vllm_wrapper import Qwen35VLLM

def main():
    # 1. 初始化模型 (全局只初始化一次)
    # 注意：如果显存不足，可以把 max_model_len 改小一点，比如 4096
    llm = Qwen35VLLM(
        model_path="/root/autodl-tmp/qwen35/tclf90/Qwen3___5-4B-AWQ",
        max_model_len=8192
    )

    # 2. 示例 1: 纯文本对话
    while True:
        user_input = input("\n请输入问题 (输入 'quit' 退出): ")
        if user_input.lower() == 'quit':
            break
            
        # 调用生成接口
        response = llm.generate(
            text=user_input,
            temperature=0.7,   # 可以在这里临时调整参数
            max_tokens=1024
        )
        
        print(f"Qwen: {response}")

    # 3. 示例 2: 带图片的推理
    # print("\n正在分析图片...")
    # answer = llm.generate(
    #     text="请详细描述这张图片。",
    #     image="path/to/your/image.jpg" # 支持图片路径
    # )
    # print(answer)

if __name__ == '__main__':
    main()