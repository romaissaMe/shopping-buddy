import redis
import os
from dotenv import load_dotenv

load_dotenv()
redis_key = os.getenv('REDIS_KEY')



redis_conn = redis.Redis(
  host='redis-12882.c259.us-central1-2.gce.cloud.redislabs.com',
  port=12882,
  password=redis_key)

print('connected to redis')