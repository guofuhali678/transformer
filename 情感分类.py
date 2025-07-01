import torch
import os
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
from transformers import DataCollatorWithPadding
from tqdm import tqdm
# 配置清华大学镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"  # 清华大学Hugging Face镜像
os.environ["HF_DATASETS_CACHE"] = r"F:/点春季学习/huggingface_cache"  # 本地缓存路径

# 从镜像站加载预训练模型和分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 加载本地IMDb数据集
dataset = load_dataset(
    "csv",
    data_files=r"F:/点春季学习/注意力机制的复现/imdb.csv",
    encoding="latin-1"  # 处理特殊字符
)["train"]

# 划分训练集和测试集
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# 分词处理
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset.set_format("torch")

# 创建数据加载器
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
train_dataloader = DataLoader(tokenized_dataset["train"], batch_size=8, shuffle=True, collate_fn=data_collator)
test_dataloader = DataLoader(tokenized_dataset["test"], batch_size=8, collate_fn=data_collator)

# 配置训练设备和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)

# 训练循环
model.train()
for epoch in range(5):
    print(f"Epoch {epoch + 1}/{2}")
    for batch in tqdm(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# 评估模型
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(test_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(batch["labels"].cpu().numpy())

# 计算准确率
accuracy = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {accuracy:.4f}")