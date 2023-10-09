import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertConfig
from sklearn.metrics import f1_score, accuracy_score

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


# 初始化多标签分类模型
num_labels = 7  # 这里假设你的模型有3个标签/4
model = MultiLabelClassifier(num_labels)

# 加载训练好的模型权重
model_path = 'incremental_multi_label_model.pth'
model.load_state_dict(torch.load(model_path))
model.eval()

# 测试数据
test_data = [
    "This is a positive sentence.",
    "This is a negative sentence.",
    "Another positive example.",
    "New positive example",
    "all",
]

# 数据预处理
max_length = 128
test_input_ids = []
test_attention_mask = []

# for text in test_data:
#     encoded_text = tokenizer.encode_plus(text, max_length=max_length, padding='max_length', truncation=True,
#                                          return_tensors='pt')
#     input_ids = encoded_text['input_ids']
#     attention_mask = encoded_text['attention_mask']
#     test_input_ids.append(input_ids)
#     test_attention_mask.append(attention_mask)
#
# test_input_ids = torch.cat(test_input_ids, dim=0)
# test_attention_mask = torch.cat(test_attention_mask, dim=0)

for text in test_data:
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

# 模型推断
with torch.no_grad():
    logits = model(test_input_ids, test_attention_mask)
    predictions = (logits > 0).float()

# 打印预测结果
print("Predictions:")
print(predictions)

# 注意：这里的 `predictions` 是一个包含多个标签的张量，你可以根据需要对其进行后续处理。
