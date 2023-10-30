import numpy as np
from redis import Redis
from redis.commands.search.field import TagField, TextField, VectorField


def load_vectors(client: Redis, product_metadata, vector_dict):
    p = client.pipeline(transaction=False)
    for index in product_metadata.keys():
        # hash key
        key = "product:" + str(index) + ":" + product_metadata[index]["primary_key"]
        # hash values
        item_metadata = product_metadata[index]
        item_keywords_vector = np.array(vector_dict[index], dtype=np.float32).tobytes()
        item_metadata["item_vector"] = item_keywords_vector
        p.hset(key, mapping=item_metadata)
    p.execute()


def create_flat_index(redis_conn, number_of_vectors, vector_dimensions=512, distance_metric="L2"):
    redis_conn.ft().create_index(
        [
            VectorField(
                "item_vector",
                "FLAT",
                {
                    "TYPE": "FLOAT32",
                    "DIM": vector_dimensions,
                    "DISTANCE_METRIC": distance_metric,
                    "INITIAL_CAP": number_of_vectors,
                    "BLOCK_SIZE": number_of_vectors,
                },
            ),
            TagField("product_type"),
            TextField("item_name"),
            TextField("item_keywords"),
            TagField("country"),
        ]
    )
