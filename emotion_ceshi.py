
#远程ssh可以是因为linux系统支持多线程，
#windows里直接num_work不兼容
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
from transformers import DataCollatorWithPadding
from tqdm import tqdm
# 加载数据集
dataset = load_dataset(
    "csv",
    data_files=r"F:/点春季学习/注意力机制的复现/imdb.csv",
    encoding="latin-1"
)
#print("数据集结构:", type(dataset))
#print("数据集拆分:", dataset.keys())  # 确认是否包含'train'拆分
dataset = dataset["train"]
#验证：
print("列名:", dataset.column_names)  # 应输出 ['row_Number', 'text', 'polarity']

dataset = dataset.train_test_split(test_size=0.2, seed=42)

model_path = r"F:/点春季学习/注意力机制的复现/huggingface"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, max_length=128)

# 处理数据集并移除无关列
tokenized_dataset = dataset.map(tokenize_function, batched=True)

columns_to_remove = ["row_Number", "text"]  
tokenized_dataset = tokenized_dataset.remove_columns(columns_to_remove)


tokenized_dataset = tokenized_dataset.rename_column("polarity", "labels")

tokenized_dataset.set_format("torch")

#验证
print("处理后的列名:", tokenized_dataset["train"].column_names)
# 应输出: ['input_ids', 'attention_mask', 'token_type_ids', 'labels']

batch_size=64
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
train_dataloader = DataLoader(tokenized_dataset["train"], 
                              batch_size=batch_size, 
                              shuffle=True, 
                              collate_fn=data_collator,
                              num_workers=4,
                              pin_memory=True
)

test_dataloader = DataLoader(
    tokenized_dataset["test"], 
    batch_size=batch_size,
    collate_fn=data_collator,
)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)




model.train()
for epoch in range(2):
    total_loss = 0
    print(f"Epoch {epoch + 1}")
    for batch in tqdm(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")

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


accuracy = accuracy_score(all_labels, all_preds)
print(f"Accuracy: {accuracy:.4f}")

