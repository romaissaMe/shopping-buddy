import os

import redis
from dotenv import load_dotenv

load_dotenv()


def create_redis():
    return redis.ConnectionPool(
        host=os.getenv("REDIS_HOST"),
        port=os.getenv("REDIS_PORT"),
        password=os.getenv("REDIS_KEY"),
        db=0,
        decode_responses=True,
    )
