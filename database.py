import redis
import os
from dotenv import load_dotenv

load_dotenv()
redis_key = os.getenv('REDIS_KEY')



redis_conn = redis.Redis(
  host='redis-10923.c10.us-east-1-4.ec2.cloud.redislabs.com',
  port=10923,
  password=redis_key)

print('connected to redis')