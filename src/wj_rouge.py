import os
import json
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from rouge_score import rouge_scorer

os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
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

# ========== ROUGE评估函数 ==========
def calculate_rouge_scores(predictions, references, rouge_types=['rouge1', 'rouge2', 'rougeL']):
    """
    计算ROUGE分数
    
    Args:
        predictions: 模型生成的文本列表
        references: 参考文本（ground truth）列表  
        rouge_types: ROUGE类型列表，默认包括['rouge1', 'rouge2', 'rougeL']
    
    Returns:
        dict: 包含各种ROUGE分数的字典
    """
    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
    
    scores = {rouge_type: {'precision': [], 'recall': [], 'fmeasure': []} for rouge_type in rouge_types}
    
    for pred, ref in zip(predictions, references):
        # 确保文本不为空
        pred = pred.strip() if pred.strip() else "empty"
        ref = ref.strip() if ref.strip() else "empty"
        
        score = scorer.score(ref, pred)
        
        for rouge_type in rouge_types:
            scores[rouge_type]['precision'].append(score[rouge_type].precision)
            scores[rouge_type]['recall'].append(score[rouge_type].recall)
            scores[rouge_type]['fmeasure'].append(score[rouge_type].fmeasure)
    
    # 计算平均值
    avg_scores = {}
    for rouge_type in rouge_types:
        avg_scores[rouge_type] = {
            'precision': sum(scores[rouge_type]['precision']) / len(scores[rouge_type]['precision']),
            'recall': sum(scores[rouge_type]['recall']) / len(scores[rouge_type]['recall']),
            'fmeasure': sum(scores[rouge_type]['fmeasure']) / len(scores[rouge_type]['fmeasure'])
        }
    
    return avg_scores

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

    # --- 加载待评估模型---
    print("正在加载待评估模型...")
    base_model_name_or_path = "Qwen/Qwen3-1.7B"
    tokenizer_target = AutoTokenizer.from_pretrained(base_model_name_or_path, use_fast=False, trust_remote_code=True)
    model_target = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    lora_checkpoint = "./output/Qwen3-1.7B/checkpoint-1084"
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


    #test_texts = test_texts[:1]  # 测试


    print(f"✅ 共加载 {len(test_texts)} 条测试样本")



    # --- 生成模型响应 ---
    print("正在生成模型响应...")
    predictions_tuned = []
    predictions_base = []

    for i, (item, gt) in enumerate(tqdm(zip(test_texts, ground_truths), total=len(test_texts), desc="生成响应")):
        try:
            messages = [
                {"role": "system", "content": item["instruction"]},
                {"role": "user", "content": item["input"]}
            ]
            # 生成微调模型的回答
            cot_response_tuned = predict(messages, model_target, tokenizer_target)
            predictions_tuned.append(cot_response_tuned)
            
            # 生成原始模型的回答
            cot_response_base = predict(messages, model_base, tokenizer_target)
            predictions_base.append(cot_response_base)

        except Exception as e:
            print(f"\n❌ 第 {i} 条样本处理失败: {e}")
            predictions_tuned.append("")
            predictions_base.append("")

    # --- 计算ROUGE分数 ---
    print("正在计算ROUGE分数...")
    rouge_types = ['rouge1', 'rouge2', 'rougeL']
    
    # 计算微调模型的ROUGE分数
    rouge_scores_tuned = calculate_rouge_scores(predictions_tuned, ground_truths, rouge_types)
    
    # 计算原始模型的ROUGE分数
    rouge_scores_base = calculate_rouge_scores(predictions_base, ground_truths, rouge_types)

    # --- 输出最终结果 ---
    print("\n" + "="*80)
    print("📊 ROUGE 评估结果:")
    print("="*80)
    
    print("-" * 80)
    print(f"模型类型        | ROUGE类型 | Precision | Recall  | F1      ")
    print("-" * 80)
    for rouge_type in rouge_types:
        # 微调模型
        print(f"微调模型 (Tuned) | {rouge_type:<8} | {rouge_scores_tuned[rouge_type]['precision']:.4f}  | {rouge_scores_tuned[rouge_type]['recall']:.4f}  | {rouge_scores_tuned[rouge_type]['fmeasure']:.4f}")
        # 原始模型
        print(f"原始模型 (Base)  | {rouge_type:<8} | {rouge_scores_base[rouge_type]['precision']:.4f}  | {rouge_scores_base[rouge_type]['recall']:.4f}  | {rouge_scores_base[rouge_type]['fmeasure']:.4f}")
        print("-" * 80)
    
    print(f"样本总数: {len(test_texts)}")
    print("="*80)

    # --- 输出性能提升 ---
    print("\n📈 性能提升分析 (Tuned - Base):")
    print("-" * 50)
    for rouge_type in rouge_types:
        improvement_precision = rouge_scores_tuned[rouge_type]['precision'] - rouge_scores_base[rouge_type]['precision']
        improvement_recall = rouge_scores_tuned[rouge_type]['recall'] - rouge_scores_base[rouge_type]['recall']
        improvement_f1 = rouge_scores_tuned[rouge_type]['fmeasure'] - rouge_scores_base[rouge_type]['fmeasure']
        print(f"  {rouge_type} - Precision: {improvement_precision:+.4f}, Recall: {improvement_recall:+.4f}, F1: {improvement_f1:+.4f}")
    
    print("="*80)
    
    # --- 可选：保存详细结果 ---
    detailed_results = {
        "rouge_scores": {
            "tuned": {k: v for k, v in rouge_scores_tuned.items()},
            "base": {k: v for k, v in rouge_scores_base.items()}
        },
        "individual_results": [
            {
                "index": i,
                "question": test_texts[i]["input"],
                "ground_truth": ground_truths[i],
                "prediction_tuned": predictions_tuned[i],
                "prediction_base": predictions_base[i]
            }
            for i in range(len(test_texts))
        ]
    }
    
    with open("rouge_evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)
    
    print("✅ 详细结果已保存至 rouge_evaluation_results.json")