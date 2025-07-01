import torch
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
from transformers import DataCollatorWithPadding
from tqdm import tqdm

# 函数定义和导入放在外面
def tokenize_function(example, tokenizer):
    return tokenizer(example["text"], truncation=True, max_length=128)

if __name__ == "__main__":
    # 加载数据集
    dataset = load_dataset(
        "csv",
        data_files=r"F:/点春季学习/注意力机制的复现/imdb.csv",
        encoding="latin-1"
    )
    dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
    
    # 验证列名
    print("列名:", dataset["train"].column_names)
    
    model_path = r"F:/点春季学习/注意力机制的复现/huggingface"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
    
    # 处理数据集
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),  # 传递tokenizer参数
        batched=True
    )
    tokenized_dataset = tokenized_dataset.remove_columns(["row_Number", "text"])
    tokenized_dataset = tokenized_dataset.rename_column("polarity", "labels")
    tokenized_dataset.set_format("torch")
    
    print("处理后的列名:", tokenized_dataset["train"].column_names)
    
    batch_size = 64
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 关键修改：设置num_workers=0或使用if __name__包裹
    train_dataloader = DataLoader(
        tokenized_dataset["train"], 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=data_collator,
        num_workers=0,  # Windows下暂时设为0避免多进程问题
        pin_memory=True
    )
    test_dataloader = DataLoader(
        tokenized_dataset["test"], 
        batch_size=batch_size,
        collate_fn=data_collator,
        num_workers=0,
        pin_memory=True
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    # 训练循环
    model.train()
    for epoch in range(5):
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