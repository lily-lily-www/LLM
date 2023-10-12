# imports
import time

import pandas as pd
import tiktoken
import openai

from openai.embeddings_utils import get_embedding
import os
import json

conf_name = "config.json"
conf_dir = os.path.dirname(os.getcwd())
conf_path = os.path.join(conf_dir, conf_name)
# 读取配置文件
with open(conf_path, "r") as file:
    config = json.load(file)
# 访问配置数据
openai.api_key = config["openapi"]["key"]

# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191

# load & inspect dataset
input_datapath = "/Users/lily/llm-model-data/Reviews.csv"  # to save space, we provide a pre-filtered dataset
df = pd.read_csv(input_datapath, index_col=0)
df = df[["Time", "ProductId", "UserId", "Score", "Summary", "Text"]]
df = df.dropna()
df["combined"] = (
        "Title: " + df.Summary.str.strip() + "; Content: " + df.Text.str.strip()
)
df.head(2)

# subsample to 1k most recent reviews and remove samples that are too long
top_n = 10
df = df.sort_values("Time").tail(
    top_n * 2)  # first cut to first 2k entries, assuming less than half will be filtered out
df.drop("Time", axis=1, inplace=True)

encoding = tiktoken.get_encoding(embedding_encoding)

# omit reviews that are too long to embed
df["n_tokens"] = df.combined.apply(lambda x: len(encoding.encode(x)))
df = df[df.n_tokens <= max_tokens].tail(top_n)
len(df)

# Ensure you have your API key set in your environment per the README: https://github.com/openai/openai-python#usage

# This may take a few minutes

# 请求次数限制
emb = []
for i in list(df['combined']):
    emb.append(get_embedding(i, engine=embedding_model))
    time.sleep(10)
df["embedding"] = emb
# df["embedding"] = df.combined.apply(lambda x: get_embedding(x, engine=embedding_model))
df.to_csv("data/fine_food_reviews_with_embeddings_1k.csv")
