import os
import json
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import snapshot_download
os.environ["CUDA_VISIBLE_DEVICES"] = "2" 
device = "cuda" if torch.cuda.is_available() else "cpu"
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

# ========== 思维链分步 ==========
import re
from typing import List

def step_partition(cot_text: str, eval_model, eval_tokenizer, max_retries=2) -> List[str]:
    prompt = [
        {"role": "system", "content": "你是一个逻辑分析专家。请将用户的推理过程拆解为若干个清晰、独立、语义完整的推理步骤。每个步骤应表达一个完整的思想或事实。请以 JSON 列表格式输出，不要包含其他内容。"},
        {"role": "user", "content": f"推理过程如下：\n\n{cot_text}\n\n请拆解为步骤（JSON 列表格式）："}
    ]

    raw_output = predict(prompt, eval_model, eval_tokenizer, max_new_tokens=1024)

    for _ in range(max_retries):
        try:
            json_match = re.search(r"\[\s*\".*?\"\s*\]", raw_output, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                steps = json.loads(json_str)
                if isinstance(steps, list) and all(isinstance(s, str) for s in steps):
                    steps = [s.strip() for s in steps if s.strip()]
                    if steps:
                        return steps
            raw_output = predict(
                prompt + [{"role": "assistant", "content": raw_output}, {"role": "user", "content": "请严格按 JSON 列表格式输出，例如：[\"步骤1\", \"步骤2\"]"}],
                eval_model,
                eval_tokenizer,
                max_new_tokens=512
            )
        except (json.JSONDecodeError, TypeError, KeyError):
            continue

    print("⚠️ LLM 分步失败，回退到规则分步")
    fallback_steps = re.split(r'\n\s*(?:\d+\.|-|\*|•)\s*', cot_text)
    fallback_steps = [s.strip() for s in fallback_steps if s.strip()]
    if len(fallback_steps) <= 1:
        fallback_steps = [s.strip() for s in cot_text.split('。') if s.strip()]
    return fallback_steps if fallback_steps else [cot_text.strip()]

# ========== 评估单个样本 ==========
def evaluate_cot_quality(question, model_response_cot, ground_truth, eval_model, eval_tokenizer):
    steps = step_partition(model_response_cot, eval_model, eval_tokenizer)
    if not steps:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0, "num_steps": 0}

    recall_hits = 0
    precision_hits = 0

    for step in steps:
        # Recall: 是否被 ground truth 支持
        recall_prompt = [
            {"role": "system", "content": "你是一个严谨的评估专家。请判断以下推理步骤的内容是否可以从真实答案中推断出（即真实答案是否支持该步骤）。只回答“是”或“否”。"},
            {"role": "user", "content": f"真实答案：{ground_truth}\n\n推理步骤：{step}"}
        ]
        recall_ans = predict(recall_prompt, eval_model, eval_tokenizer, max_new_tokens=10).strip()
        is_supported = "是" in recall_ans

        # Precision: 步骤本身是否正确
        prec_prompt = [
            {"role": "system", "content": "你是一个严谨的评估专家。请判断以下推理步骤在事实和逻辑上是否正确。只回答“正确”或“错误”。"},
            {"role": "user", "content": f"问题：{question}\n\n推理步骤：{step}"}
        ]
        prec_ans = predict(prec_prompt, eval_model, eval_tokenizer, max_new_tokens=10).strip()
        is_correct = "正确" in prec_ans

        if is_supported:
            recall_hits += 1
        if is_correct:
            precision_hits += 1

    total = len(steps)
    recall = recall_hits / total if total > 0 else 0
    precision = precision_hits / total if total > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "num_steps": total
    }

# ========== 数据转换函数（模拟）==========
def dataset_jsonl_transfer(src_path, dst_path):
    """模拟格式转换：将原始 test.jsonl 转为 {instruction, input, output} 格式"""
    try:
        with open(src_path, "r", encoding="utf-8") as f_in, open(dst_path, "w", encoding="utf-8") as f_out:
            for line in f_in:
                data = json.loads(line.strip())
                # 假设原始格式是 {'question': ..., 'answer': ..., 'cot': ...}
                # 这里按你的训练格式统一为 instruction + input + output
                item = {
                    "instruction": "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。",
                    "input": data.get("question", ""),
                    "output": data.get("answer", "")
                }
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
        return True
    except Exception as e:
        print(f"转换失败: {e}")
        return False

# ========== 主程序 ==========
if __name__ == "__main__":
    # --- 配置路径 ---
    test_dataset_path = "./test.jsonl"
    test_format_path = "./test_format.jsonl"

    # --- 加载待评估模型（LoRA 微调）---
    print("正在加载待评估模型（LoRA 微调版）...")
    base_model_name_or_path = "Qwen/Qwen3-1.7B"
    tokenizer_target = AutoTokenizer.from_pretrained(base_model_name_or_path, use_fast=False, trust_remote_code=True)
    model_target = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    lora_checkpoint = "./output/Qwen3-1.7B-lora/checkpoint-1084"
    model_target = PeftModel.from_pretrained(model_target, lora_checkpoint)
    model_target = model_target.merge_and_unload()

    # --- 加载原始基础模型（用于对比）---
    print("正在加载原始基础模型（Qwen3-1.7B）...")
    model_base = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    # --- 加载评估模型（Qwen2-7B-Instruct）---
    # --- 加载评估模型（Qwen2-7B-Instruct）---
    print("正在加载评估模型...")
    #eval_model_path = "Qwen/Qwen2-7B-Instruct"  # 直接使用模型ID
    
    eval_model_path = "./models--Qwen--Qwen2-7B-Instruct/snapshots/f2826a00ceef68f0f2b946d945ecc0477ce4450c"
    try:
        tokenizer_eval = AutoTokenizer.from_pretrained(eval_model_path, use_fast=False, trust_remote_code=True)
        model_eval = AutoModelForCausalLM.from_pretrained(
            eval_model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
    except Exception as e:
        print("❌ 评估模型 Qwen2-7B-Instruct 加载失败")
        print("错误:", e)
        exit(1)

    


    # --- 准备测试数据 ---
    if os.path.exists(test_dataset_path):
        print("正在转换测试数据集格式...")
        if not os.path.exists(test_format_path):
            if not dataset_jsonl_transfer(test_dataset_path, test_format_path):
                print("⚠️ 转换失败，使用默认测试数据")
                test_texts, ground_truths = None, None
            else:
                print(f"✅ 转换成功，保存至 {test_format_path}")
        else:
            print(f"✅ 使用已存在的格式化数据: {test_format_path}")

        if os.path.exists(test_format_path):
            test_df = pd.read_json(test_format_path, lines=True)
            test_texts = [
                {"instruction": row["instruction"], "input": row["input"]}
                for _, row in test_df.iterrows()
            ]
            ground_truths = test_df["output"].tolist()
        else:
            test_texts = None
    else:
        test_texts = None

    # --- Fallback to default test data ---
    if test_texts is None or len(test_texts) == 0:
        print("⚠️ 使用默认测试数据")
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
        ground_truths = [
            "糖尿病患者应选择低升糖指数（GI）的碳水化合物，如全谷物、豆类、蔬菜等，避免精制糖和白面包等高GI食物，以帮助控制血糖水平。",
            "抗溃疡药物主要包括质子泵抑制剂（如奥美拉唑）、H2受体拮抗剂（如雷尼替丁）和胃黏膜保护剂（如硫糖铝）。它们通过不同机制减少胃酸分泌或保护胃黏膜，维持胃黏膜的保护与损伤平衡。"
        ]

    #test_texts = test_texts[:10]
    #ground_truths = ground_truths[:10]

    print(f"✅ 共加载 {len(test_texts)} 条测试样本")

    # --- 开始批量评估 ---
    all_results_tuned = []
    all_results_base = []
    for i, (item, gt) in enumerate(tqdm(zip(test_texts, ground_truths), total=len(test_texts), desc="评估进度")):
        try:
            messages = [
                {"role": "system", "content": item["instruction"]},
                {"role": "user", "content": item["input"]}
            ]
            # 生成微调模型的回答
            cot_response_tuned = predict(messages, model_target, tokenizer_target)
            # 生成原始模型的回答
            cot_response_base = predict(messages, model_base, tokenizer_target)  # 注意：tokenizer 用同一个

            # 评估微调模型的回答
            result_tuned = evaluate_cot_quality(
                question=item["input"],
                model_response_cot=cot_response_tuned,
                ground_truth=gt,
                eval_model=model_eval,
                eval_tokenizer=tokenizer_eval
            )
            all_results_tuned.append(result_tuned)

            # 评估原始模型的回答
            result_base = evaluate_cot_quality(
                question=item["input"],
                model_response_cot=cot_response_base,
                ground_truth=gt,
                eval_model=model_eval,
                eval_tokenizer=tokenizer_eval
            )
            all_results_base.append(result_base)

        except Exception as e:
            print(f"\n❌ 第 {i} 条样本评估失败: {e}")
            all_results_tuned.append({"recall": 0.0, "precision": 0.0, "f1": 0.0, "num_steps": 0})
            all_results_base.append({"recall": 0.0, "precision": 0.0, "f1": 0.0, "num_steps": 0})

    # --- 计算平均指标 ---
    total = len(test_texts)

    # 微调模型指标
    avg_recall_tuned = sum(r["recall"] for r in all_results_tuned) / total if total > 0 else 0
    avg_precision_tuned = sum(r["precision"] for r in all_results_tuned) / total if total > 0 else 0
    avg_f1_tuned = sum(r["f1"] for r in all_results_tuned) / total if total > 0 else 0
    avg_steps_tuned = sum(r["num_steps"] for r in all_results_tuned) / total if total > 0 else 0

    # 原始模型指标
    avg_recall_base = sum(r["recall"] for r in all_results_base) / total if total > 0 else 0
    avg_precision_base = sum(r["precision"] for r in all_results_base) / total if total > 0 else 0
    avg_f1_base = sum(r["f1"] for r in all_results_base) / total if total > 0 else 0
    avg_steps_base = sum(r["num_steps"] for r in all_results_base) / total if total > 0 else 0

    # --- 输出最终结果 ---
    print("\n" + "="*80)
    print("📊 最终评估结果（平均值）:")
    print("-" * 80)
    print(f"模型类型        | Recall  | Precision | F1      | Avg Steps")
    print("-" * 80)
    print(f"微调模型 (Tuned) | {avg_recall_tuned:.4f}  | {avg_precision_tuned:.4f}     | {avg_f1_tuned:.4f}  | {avg_steps_tuned:.2f}")
    print(f"原始模型 (Base)  | {avg_recall_base:.4f}  | {avg_precision_base:.4f}     | {avg_f1_base:.4f}  | {avg_steps_base:.2f}")
    print("-" * 80)
    print(f"样本总数: {total}")
    print("="*80)

    # --- 可选：输出性能提升 ---
    improvement_recall = avg_recall_tuned - avg_recall_base
    improvement_precision = avg_precision_tuned - avg_precision_base
    improvement_f1 = avg_f1_tuned - avg_f1_base
    print(f"📈 性能提升 (Tuned - Base):")
    print(f"  Recall:    {improvement_recall:+.4f}")
    print(f"  Precision: {improvement_precision:+.4f}")
    print(f"  F1:        {improvement_f1:+.4f}")
    print("="*80)
