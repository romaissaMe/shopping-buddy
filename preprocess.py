import numpy as np
import pandas as pd
import redis
from sentence_transformers import SentenceTransformer

from constants import (
    DATA_PATH,
    MAX_TEXT_LENGTH,
    NUMBER_PRODUCTS,
    TEXT_EMBEDDING_DIMENSION,
)
from database import create_redis
from utils import create_flat_index, load_vectors


def auto_truncate(text: str):
    return text[0:MAX_TEXT_LENGTH]


def data_preprocessing_and_loading():
    pool = create_redis()
    redis_conn = redis.Redis(connection_pool=pool)
    data = pd.read_csv(
        DATA_PATH,
        converters={"bullet_point": auto_truncate, "item_keywords": auto_truncate, "item_name": auto_truncate},
    )
    data["primary_key"] = data["item_id"] + "-" + data["domain_name"]
    data.drop(columns=["item_id", "domain_name"], inplace=True)
    data["item_keywords"].replace("", np.nan, inplace=True)
    data.dropna(subset=["item_keywords"], inplace=True)
    data.reset_index(drop=True, inplace=True)
    data_metadata = data.head(NUMBER_PRODUCTS).to_dict(orient="index")

    # generate embeddings (vectors) for the item keywords
    embedding_model = SentenceTransformer("sentence-transformers/all-distilroberta-v1")
    # get the item keywords attribute for each product and encode them into vector embeddings
    item_keywords = [data_metadata[i]["item_keywords"] for i in data_metadata.keys()]
    item_keywords_vectors = [embedding_model.encode(item) for item in item_keywords]
    # flush all data
    redis_conn.flushall()
    # create flat index & load vectors
    create_flat_index(redis_conn, NUMBER_PRODUCTS, TEXT_EMBEDDING_DIMENSION, "COSINE")
    load_vectors(redis_conn, data_metadata, item_keywords_vectors)


if __name__ == "__main__":
    data_preprocessing_and_loading()
