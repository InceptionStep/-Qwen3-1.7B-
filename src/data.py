from modelscope.msdatasets import MsDataset
import json
import random

# 设置随机种子以确保可重复性
random.seed(42)

# 加载数据集
ds = MsDataset.load('krisfu/delicate_medical_r1_data', subset_name='default', split='train')

# 将数据集转换为列表
data_list = list(ds)

# 随机打乱数据
random.shuffle(data_list)

# 计算分割点 (80% 训练, 10% 验证, 10% 测试)
total_size = len(data_list)
train_split_idx = int(total_size * 0.8)
val_split_idx = int(total_size * 0.9)

# 分割数据
train_data = data_list[:train_split_idx]
val_data = data_list[train_split_idx:val_split_idx]
test_data = data_list[val_split_idx:]

# 保存训练集
with open('train.jsonl', 'w', encoding='utf-8') as f:
    for item in train_data:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')

# 保存验证集
with open('val.jsonl', 'w', encoding='utf-8') as f:
    for item in val_data:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')

# 保存测试集
with open('test.jsonl', 'w', encoding='utf-8') as f:
    for item in test_data:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')

print(f"数据集已分割完成：")
print(f"训练集大小：{len(train_data)} ({len(train_data)/total_size*100:.1f}%)")
print(f"验证集大小：{len(val_data)} ({len(val_data)/total_size*100:.1f}%)")
print(f"测试集大小：{len(test_data)} ({len(test_data)/total_size*100:.1f}%)")