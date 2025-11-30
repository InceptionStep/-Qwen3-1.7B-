import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ========== 推理函数（通用）==========
def predict(messages, model, tokenizer, max_new_tokens=2048):
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()

# ========== 主程序：仅运行 Base 模型 ==========
if __name__ == "__main__":
    # 加载 Base 模型（Qwen3-1.7B）
    base_model_name_or_path = "Qwen/Qwen3-1.7B"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    # 测试输入
    instruction = "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。"
    # instruction = "医生，我听说凝血块机化后会对呼吸功能造成损害，这是真的吗？具体是怎么影响的呢？而且，这种影响在不同类型的血胸中有什么不同吗？"
    input_value = "医生，我听说凝血块机化后会对呼吸功能造成损害，这是真的吗？具体是怎么影响的呢？而且，这种影响在不同类型的血胸中有什么不同吗？"

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": input_value}
    ]

    # 仅运行 Base 模型推理
    base_response = predict(messages, model, tokenizer)
    print("🟨 原始基础模型生成的回答：")
    print(base_response)