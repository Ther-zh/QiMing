# -*- coding: utf-8 -*-
import ollama

response = ollama.chat(model='qwen3.5-4b', messages=[
    {'role': 'user', 'content': '请介绍一下你自己，并简单描述一下你能做什么？'}
])
print("="*50)
print("模型回复:")
print("="*50)
print(response['message']['content'])
print("="*50)
