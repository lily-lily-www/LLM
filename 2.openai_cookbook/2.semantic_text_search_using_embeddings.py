import pandas as pd
import numpy as np
from ast import literal_eval
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
#1. load data
datafile_path = "/Users/lily/gitcode/LLM/2.openai_cookbook/data/fine_food_reviews_with_embeddings_1k.csv"
df = pd.read_csv(datafile_path)
#2. get embedding
df["embedding"] = df.embedding.apply(literal_eval).apply(np.array)


from openai.embeddings_utils import get_embedding, cosine_similarity

# search through the reviews for a specific product
def search_reviews(df, product_description, n=3, pprint=True):
    product_embedding = get_embedding(
        product_description,
        engine="text-embedding-ada-002"
    )

    df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, product_embedding))

    results = (
        df.sort_values("similarity", ascending=False)
        .head(n)
        .combined.str.replace("Title: ", "")
        .str.replace("; Content:", ": ")
    )
    if pprint:
        for r in results:
            print(r[:200])
            print()
    return results


results = search_reviews(df, "delicious beans", n=3)
