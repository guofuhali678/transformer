#不能有大写字母
from transformers import BertTokenizer
import re
import string
from collections import OrderedDict
model_path = "F:\点春季学习\注意力机制的复现\huggingface"  
# 从本地加载分词器
tokenizer = BertTokenizer.from_pretrained(model_path)
text = 'the peach tree beams so red,how brilliant are its flowers.the maid getting wed,good for the naptial bowers'
text=text.lower()
text=re.sub(r'\d+','',text)
chinese_punctual=r'[！？：；“”。，（）》《【】]'
text=re.sub(chinese_punctual,'',text)
text=re.sub(r'\s+','',text).strip()
words=text.split()
unique_words=list(OrderedDict.fromkeys(words))
cleaned_text=' '.join(unique_words)
encoding = tokenizer.encode(cleaned_text)
print("Tokenizer IDs:", encoding)
tokens = tokenizer.convert_ids_to_tokens(encoding)
print("Tokens:", tokens)

