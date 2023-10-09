import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel, BertConfig
from sklearn.metrics import f1_score, accuracy_score

# # 示例多标签分类数据
old_data = [
    ("This is a positive sentence.", [1, 0, 1]),
    ("This is a negative sentence.", [0, 1, 1]),
    ("Another positive example.", [1, 0, 1]),
    ("Another negative example.", [0, 1, 1]),
    ("nothing.", [0, 0, 0]),
    ("all", [1, 1, 0]),
    ("no", [0, 0, 0]),
    ("maybe", [1, 1, 0])
]
# # 将数据分为训练集和测试集
# texts, labels = zip(*data)
# train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# 加载预训练的BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
config = BertConfig.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased', config=config)

#
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


# 初始化多标签分类模型
# num_labels = len(train_labels[0])
num_labels = 3
model = MultiLabelClassifier(num_labels)

# 加载训练好的模型权重
model_path = 'multi_label_model.pth'
model.load_state_dict(torch.load(model_path))

# 新增标签数据
add_data = [
    ("New positive example.", [1, 0, 1, 1]),
    ("New negative example.", [0, 1, 1, 1]),
    ("Another new positive example.", [1, 0, 1, 1]),
    ("Another new negative example.", [0, 1, 1, 1])
]
add_data_with_zeros = [(text, [0, 0, 0] + labels) for text, labels in add_data]
old_data_with_zeros = [(text, labels + [0, 0, 0, 0]) for text, labels in old_data]

new_data = add_data_with_zeros + old_data_with_zeros
# 将数据分为训练集和测试集
new_texts, new_labels = zip(*new_data)
new_train_texts, new_test_texts, new_train_labels, new_test_labels = train_test_split(new_texts, new_labels,
                                                                                      test_size=0.2, random_state=42)

# 数据预处理（与之前相同）
max_length = 128
new_train_input_ids = []
new_train_attention_mask = []

# 统一所有文本的长度
for text in new_train_texts:
    encoded_text = tokenizer.encode_plus(text, max_length=max_length, padding='max_length', truncation=True,
                                         return_tensors='pt')
    input_ids = encoded_text['input_ids']
    attention_mask = encoded_text['attention_mask']

    # 将 input_ids 和 attention_mask 调整为相同的大小
    padding_length = max_length - input_ids.size(1)
    input_ids = torch.cat([input_ids, torch.zeros((input_ids.size(0), padding_length), dtype=torch.long)], dim=1)
    attention_mask = torch.cat(
        [attention_mask, torch.zeros((attention_mask.size(0), padding_length), dtype=torch.long)], dim=1)

    new_train_input_ids.append(input_ids)
    new_train_attention_mask.append(attention_mask)

# 确保 train_input_ids 和 train_attention_mask 中的张量大小一致
new_train_input_ids = torch.cat(new_train_input_ids, dim=0)
new_train_attention_mask = torch.cat(new_train_attention_mask, dim=0)
new_train_labels = torch.tensor(new_train_labels, dtype=torch.float32)
# 定义新的标签分类层（增加了一个新的标签）
# num_new_labels = len(new_data[0][1])  # 新增标签的数量
# num_current_labels = num_labels + num_new_labels
# model.classifier = nn.Linear(model.bert.config.hidden_size, num_current_labels)


# 创建新增分类器层
new_labels = len(new_data[0][1])
new_classifier = nn.Linear(bert_model.config.hidden_size, new_labels)
# 将新增分类器添加到原有模型之后，构建新模型
# new_model = nn.Sequential(
#     model.bert,  # 使用原始模型的bert部分
#     new_classifie r  # 新增分类器
# )
# # 将新增分类器添加到原有模型之后，构建新模型
# new_model = nn.Sequential(
#     model,  # 原有模型
#     new_classifier  # 新增分类器
# )
# new_bert = model.bert
# new_model = nn.Module()
# new_model.bert = new_bert
# new_model.classifier = new_classifier
model.classifier = new_classifier
new_model = model
# 冻结原有模型的权重，只训练新增分类器
for param in new_model.bert.parameters():
    param.requires_grad = False

# 损失函数和优化器（与之前相同）
criterion = nn.BCEWithLogitsLoss()  # 二进制交叉熵损失
optimizer = torch.optim.Adam(new_model.classifier.parameters(), lr=2e-5)

# 增量训练
epochs = 5
batch_size = 2

for epoch in range(epochs):
    new_model.train()
    for i in range(0, len(new_train_input_ids), batch_size):
        optimizer.zero_grad()
        batch_input_ids = new_train_input_ids[i:i + batch_size]
        batch_attention_mask = new_train_attention_mask[i:i + batch_size]
        batch_labels = new_train_labels[i:i + batch_size]
        # logits = new_model(batch_input_ids, batch_attention_mask)
        logits = new_model(batch_input_ids, batch_attention_mask)
        # 计算损失
        loss = criterion(logits, batch_labels)
        loss.backward()
        optimizer.step()

# 模型评估（与之前相同）
new_test_input_ids = []
new_test_attention_mask = []

for text in new_test_texts:
    encoded_text = tokenizer.encode_plus(text, max_length=max_length, padding='max_length', truncation=True,
                                         return_tensors='pt')
    input_ids = encoded_text['input_ids']
    attention_mask = encoded_text['attention_mask']

    # 将 input_ids 和 attention_mask 调整为相同的大小
    padding_length = max_length - input_ids.size(1)
    input_ids = torch.cat([input_ids, torch.zeros((input_ids.size(0), padding_length), dtype=torch.long)], dim=1)
    attention_mask = torch.cat(
        [attention_mask, torch.zeros((attention_mask.size(0), padding_length), dtype=torch.long)], dim=1)

    new_test_input_ids.append(input_ids)
    new_test_attention_mask.append(attention_mask)
new_test_input_ids = torch.cat(new_test_input_ids, dim=0)
new_test_attention_mask = torch.cat(new_test_attention_mask, dim=0)
new_test_labels = torch.tensor(new_test_labels, dtype=torch.float32)

with torch.no_grad():
    new_model.eval()
    logits = new_model(new_test_input_ids, new_test_attention_mask)
    predictions = (logits > 0).float()
# 保存增量训练后的模型
model_path = 'incremental_multi_label_model.pth'
torch.save(new_model.state_dict(), model_path)
print(f"Incrementally trained model saved to {model_path}")
