# -*- coding: utf-8 -*-
import ollama

# 测试1: 数学推理
print("="*60)
print("测试1: 数学推理能力")
print("="*60)
response1 = ollama.chat(model='qwen3.5-4b', messages=[
    {'role': 'user', 'content': '如果一个苹果5元，买3个苹果需要多少钱？请一步步思考。'}
])
print(response1['message']['content'])
print()

# 测试2: 代码能力
print("="*60)
print("测试2: 代码生成能力")
print("="*60)
response2 = ollama.chat(model='qwen3.5-4b', messages=[
    {'role': 'user', 'content': '请用Python写一个计算斐波那契数列的函数，并加上注释。'}
])
print(response2['message']['content'])
print()

# 测试3: 中文理解
print("="*60)
print("测试3: 中文理解与创作")
print("="*60)
response3 = ollama.chat(model='qwen3.5-4b', messages=[
    {'role': 'user', 'content': '请写一首关于春天的五言绝句。'}
])
print(response3['message']['content'])
print()

print("="*60)
print("所有测试完成！")
print("="*60)
