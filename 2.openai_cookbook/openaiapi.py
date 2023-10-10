import time
import openai
import json
import os
conf_name = "config.json"
conf_dir = os.path.dirname(os.getcwd())
conf_path = os.path.join(conf_dir, conf_name)
# 读取配置文件
with open(conf_path, "r") as file:
    config = json.load(file)

# 访问配置数据
openai.api_key = config["openapi"]["key"]

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": 'how are you'}
    ],
    temperature=0,
    max_tokens=1000,
    stream=True,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    user='RdFast智能创作机器人小程序'
)

print(response)
print('response["choices"][0]["text"]结果如下所示：')
ans = ''
for r in response:
    if 'content' in r.choices[0].delta:
        ans += r.choices[0].delta['content']
        print(ans)

print(ans)
