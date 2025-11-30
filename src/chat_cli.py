import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse

import readline  # 命令行输入更友好（上下箭头历史记录）


# -----------------------------
# 预测函数（与你项目保持一致）
# -----------------------------
def predict(messages, model, tokenizer, device, max_new_tokens=512):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=max_new_tokens,
    )

    # 去掉输入部分，仅保留模型输出
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


# -----------------------------
# 聊天主程序
# -----------------------------
def chat(model_path, lora_path=None, system_prompt="你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。"):
    # 自动选择设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"加载设备: {device}")

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)

    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # 如需 LoRA
    if lora_path:
        print(f"加载 LoRA 权重: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)

    print("模型加载完成，进入对话模式（输入 exit 或 Ctrl+C 退出）\n")

    # -------------------
    # 多轮对话 messages
    # -------------------
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    # -------------------
    # 开始循环对话
    # -------------------
    while True:
        try:
            user_input = input("用户: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("结束对话")
                break

            # 将这轮加入上下文
            messages.append({"role": "user", "content": user_input})

            # 调用推理
            response = predict(messages, model, tokenizer, device)

            print("\n模型:", response, "\n")

            # 将回答也加入上下文，便于多轮对话
            messages.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            print("\n退出")
            break


# ---------------------------
# CLI 参数
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="基础模型 / 微调模型 checkpoint 路径"
    )

    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="LoRA checkpoint 路径（可选）"
    )

    args = parser.parse_args()

    chat(args.model_path, args.lora_path)
