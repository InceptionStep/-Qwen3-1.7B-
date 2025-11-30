import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
device = "cuda" if torch.cuda.is_available() else "cpu"
def predict(messages, model, tokenizer):
    # if torch.backends.mps.is_available():
    #     device = "mps"
    # elif torch.cuda.is_available():
    #     device = "cuda"
    # else:
    #     device = "cpu"

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=2048)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


# 定义要测试的 checkpoint 路径
checkpoint_paths = [
    # "./Qwen/Qwen3-1.7B",  # 原始模型（如果保存了的话）
    "./output/Qwen3-1.7B/checkpoint-0",  # 微调前的模型
    "./output/Qwen3-1.7B/checkpoint-400",
    "./output/Qwen3-1.7B/checkpoint-800",
    "./output/Qwen3-1.7B/checkpoint-1084",  # 最终模型
]

test_texts = [
    {
        'instruction': "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。",
        'input': "医生，我最近被诊断为糖尿病，听说碳水化合物的选择很重要，我应该选择什么样的碳水化合物呢？"
    },
    {
        'instruction': "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。",
        'input': "医生，我最近胃部不适，听说有几种抗溃疡药物可以治疗，您能详细介绍一下这些药物的分类、作用机制以及它们是如何影响胃黏膜的保护与损伤平衡的吗？"
    }
]

# 测试每个 checkpoint
for checkpoint_path in checkpoint_paths:
    print(f"\n{'='*50}")
    print(f"测试模型: {checkpoint_path}")
    print(f"{'='*50}")
    
    try:
        # 加载模型和 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=False, trust_remote_code=True, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path, device_map="auto", torch_dtype=torch.bfloat16, local_files_only=True)
        
        # 测试所有问题
        for idx, item in enumerate(test_texts):
            instruction = item['instruction']
            input_value = item['input']

            messages = [
                {"role": "system", "content": f"{instruction}"},
                {"role": "user", "content": f"{input_value}"}
            ]

            response = predict(messages, model, tokenizer)
            print(f"\n问题 {idx + 1}: {input_value}")
            print(f"回答: {response}")
            
    except Exception as e:
        print(f"加载 {checkpoint_path} 时出错: {e}")