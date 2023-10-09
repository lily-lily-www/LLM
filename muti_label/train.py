import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

# 示例多标签分类数据
data = [
    ("This is a positive sentence.", [1, 0, 1]),
    ("This is a negative sentence.", [0, 1, 1]),
    ("Another positive example.", [1, 0, 1]),
    ("Another negative example.", [0, 1, 1]),
    ("nothing.", [0, 0, 0]),
    ("all", [1, 1, 0]),
    ("no", [0, 0, 0]),
    ("maybe", [1, 1, 0])
]

# 将数据分为训练集和测试集
texts, labels = zip(*data)
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# 加载预训练的BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
config = BertConfig.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased', config=config)


# 自定义多标签分类模型
class MultiLabelClassifier(nn.Module):
    def __init__(self, num_labels):
        super(MultiLabelClassifier, self).__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[0][:, 0]
        logits = self.classifier(pooled_output)
        return logits


# 数据预处理
max_length = 128
train_input_ids = []
train_attention_mask = []

# 统一所有文本的长度
for text in train_texts:
    encoded_text = tokenizer.encode_plus(text, max_length=max_length, padding='max_length', truncation=True,
                                         return_tensors='pt')
    input_ids = encoded_text['input_ids']
    attention_mask = encoded_text['attention_mask']

    # 将 input_ids 和 attention_mask 调整为相同的大小
    padding_length = max_length - input_ids.size(1)
    input_ids = torch.cat([input_ids, torch.zeros((input_ids.size(0), padding_length), dtype=torch.long)], dim=1)
    attention_mask = torch.cat(
        [attention_mask, torch.zeros((attention_mask.size(0), padding_length), dtype=torch.long)], dim=1)

    train_input_ids.append(input_ids)
    train_attention_mask.append(attention_mask)

# 确保 train_input_ids 和 train_attention_mask 中的张量大小一致
train_input_ids = torch.cat(train_input_ids, dim=0)
train_attention_mask = torch.cat(train_attention_mask, dim=0)
train_labels = torch.tensor(train_labels, dtype=torch.float32)

# 初始化多标签分类模型
num_labels = len(train_labels[0])
model = MultiLabelClassifier(num_labels)

# 损失函数和优化器
criterion = nn.BCEWithLogitsLoss()  # 二进制交叉熵损失
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

# 训练模型
epochs = 5
batch_size = 2

for epoch in range(epochs):
    model.train()
    for i in range(0, len(train_input_ids), batch_size):
        optimizer.zero_grad()
        batch_input_ids = train_input_ids[i:i + batch_size]
        batch_attention_mask = train_attention_mask[i:i + batch_size]
        batch_labels = train_labels[i:i + batch_size]
        logits = model(batch_input_ids, batch_attention_mask)
        loss = criterion(logits, batch_labels)
        loss.backward()
        optimizer.step()

# 模型评估
test_input_ids = []
test_attention_mask = []

for text in test_texts:
    encoded_text = tokenizer.encode_plus(text, max_length=max_length, padding='max_length', truncation=True,
                                         return_tensors='pt')
    input_ids = encoded_text['input_ids']
    attention_mask = encoded_text['attention_mask']

    # 将 input_ids 和 attention_mask 调整为相同的大小
    padding_length = max_length - input_ids.size(1)
    input_ids = torch.cat([input_ids, torch.zeros((input_ids.size(0), padding_length), dtype=torch.long)], dim=1)
    attention_mask = torch.cat(
        [attention_mask, torch.zeros((attention_mask.size(0), padding_length), dtype=torch.long)], dim=1)

    test_input_ids.append(input_ids)
    test_attention_mask.append(attention_mask)
test_input_ids = torch.cat(test_input_ids, dim=0)
test_attention_mask = torch.cat(test_attention_mask, dim=0)
test_labels = torch.tensor(test_labels, dtype=torch.float32)

with torch.no_grad():
    model.eval()
    logits = model(test_input_ids, test_attention_mask)
    predictions = (logits > 0).float()
    # f1 = f1_score(test_labels, predictions, average='micro')
    # accuracy = accuracy_score(test_labels, predictions)

model_path = 'multi_label_model.pth'
torch.save(model.state_dict(), model_path)
# print(f"F1 Score: {f1}")
# print(f"Accuracy: {accuracy}")


"""
--------------新增标签-------------法2：增量训练
"""
