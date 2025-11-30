# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import PeftModel
# import gradio as gr
#
# # -----------------------------
# # 预测函数（与你项目保持一致）
# # -----------------------------
# def predict(messages, model, tokenizer, device, max_new_tokens=512):
#     text = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True
#     )
#
#     model_inputs = tokenizer([text], return_tensors="pt").to(device)
#
#     generated_ids = model.generate(
#         model_inputs.input_ids,
#         max_new_tokens=max_new_tokens,
#     )
#
#     # 去掉输入部分，仅保留输出
#     generated_ids = [
#         output_ids[len(input_ids):]
#         for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
#     ]
#
#     response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
#     return response
#
#
# # -----------------------------
# # 创建 Gradio 聊天界面
# # -----------------------------
# def create_chat_interface(model_path, lora_path=None):
#
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"加载设备: {device}")
#
#     # 加载 tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
#
#     # 加载基础模型
#     model = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         torch_dtype=torch.bfloat16,
#         device_map="auto"
#     )
#
#     # LoRA（可选）
#     if lora_path:
#         print(f"加载 LoRA 权重: {lora_path}")
#         model = PeftModel.from_pretrained(model, lora_path)
#
#     messages = [{"role": "system", "content": "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。"}]
#
#     # -------------------
#     # Gradio 回调函数
#     # -------------------
#     def chat_fn(user_input, chat_history):
#         messages.append({"role": "user", "content": user_input})
#
#         response = predict(messages, model, tokenizer, device)
#
#         messages.append({"role": "assistant", "content": response})
#         chat_history.append((user_input, response))
#
#         return chat_history, ""
#
#     # -------------------
#     # 构建界面
#     # -------------------
#     with gr.Blocks(title="医学对话模型 Demo") as demo:
#         gr.Markdown(
#             """
#             # 🩺 医学大模型 Demo
#             #### 支持多轮对话、LoRA 微调、医学问答展示
#             """
#         )
#
#         chatbot = gr.Chatbot(
#             height=450,
#             label="聊天窗口"
#         )
#
#         with gr.Row():
#             user_input = gr.Textbox(
#                 placeholder="请输入你的问题…",
#                 scale=5
#             )
#             submit_btn = gr.Button("发送", scale=1)
#
#         submit_btn.click(
#             fn=chat_fn,
#             inputs=[user_input, chatbot],
#             outputs=[chatbot, user_input]
#         )
#
#         user_input.submit(
#             fn=chat_fn,
#             inputs=[user_input, chatbot],
#             outputs=[chatbot, user_input]
#         )
#
#     return demo
#
#
# # -----------------------------
# # 主入口
# # -----------------------------
# if __name__ == "__main__":
#     import argparse
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_path", type=str, required=True)
#     parser.add_argument("--lora_path", type=str, default=None)
#     parser.add_argument("--port", type=int, default=7860)
#
#     args = parser.parse_args()
#
#     demo = create_chat_interface(args.model_path, args.lora_path)
#     demo.launch(server_name="0.0.0.0", server_port=args.port, share=True)
#
#
#
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import gradio as gr
import re


# -----------------------------
# 预测 + 分离 think / answer
# -----------------------------
def predict(messages, model, tokenizer, device, max_new_tokens=1024):
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

    # 去掉输入 prompt
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    full_output = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    # -----------------------------
    # 分离 think 与 answer
    # -----------------------------
    think_pattern = r"<think>(.*?)</think>"
    think_match = re.search(think_pattern, full_output, re.S)

    if think_match:
        think = think_match.group(1).strip()
        # 最终输出去掉 think 块
        answer = re.sub(think_pattern, "", full_output, flags=re.S).strip()
    else:
        think = ""
        answer = full_output.strip()

    return think, answer


# -----------------------------
# 创建 Gradio 聊天界面
# -----------------------------
def create_chat_interface(model_path, lora_path=None):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"加载设备: {device}")

    # 修改这一行
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True, local_files_only=True)

    # 修改这一行
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True
    )
    

    if lora_path:
        print(f"加载 LoRA 权重: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)

    messages = [
        {"role": "system", "content": "你是一个医学专家，你需要根据用户的问题，给出带有思考（think）的回答。"}
    ]

    # -------------------
    # Gradio 回调
    # -------------------
    def chat_fn(user_input, chat_history, think_box):
        messages.append({"role": "user", "content": user_input})

        think, answer = predict(messages, model, tokenizer, device)

        messages.append({"role": "assistant", "content": answer})

        chat_history.append((user_input, answer))

        return chat_history, "", think

    # -------------------
    # 构建界面
    # -------------------
    with gr.Blocks(title="医学大模型 Qwen3-Medical-SFT") as demo:
        gr.Markdown(
            """
            # 🩺 医学大模型 Qwen3-Medical-SFT
            ### ✔ 支持多轮对话  
            ### ✔ LoRA 微调  
            """
        )

        with gr.Row():
            chatbot = gr.Chatbot(
                height=450,
                label="模型回答（Answer）"
            )
            think_box = gr.Textbox(
                label="模型思考过程（Think）",
                placeholder="模型的 <think> 思考过程将在这里显示…",
                lines=20
            )

        with gr.Row():
            user_input = gr.Textbox(
                placeholder="请输入你的问题…",
                scale=5
            )
            submit_btn = gr.Button("发送", scale=1)

        submit_btn.click(
            fn=chat_fn,
            inputs=[user_input, chatbot, think_box],
            outputs=[chatbot, user_input, think_box]
        )

        user_input.submit(
            fn=chat_fn,
            inputs=[user_input, chatbot, think_box],
            outputs=[chatbot, user_input, think_box]
        )

    return demo


# -----------------------------
# 主入口
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--port", type=int, default=7860)

    args = parser.parse_args()

    demo = create_chat_interface(args.model_path, args.lora_path)
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=True)
