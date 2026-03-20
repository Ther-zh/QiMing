from qwen35 import Qwen35Ollama

def main():
    llm = Qwen35Ollama(model_name="qwen3.5-4b")

    while True:
        user_input = input("\n请输入问题 (输入 'quit' 退出): ")
        if user_input.lower() == 'quit':
            break
            
        response = llm.generate(
            text=user_input,
            temperature=0.7,
            max_tokens=1024
        )
        
        print(f"Qwen: {response}")

if __name__ == '__main__':
    main()
