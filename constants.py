import os

from dotenv import load_dotenv

load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
FALCON_REPO_ID = "tiiuae/falcon-7b-instruct"
FALCON_TEMPERATURE = 0.1
FALCON_MAX_TOKENS = 500

OPENAI_MODEL_NAME = "gpt-3.5-turbo"
OPENAI_TEMPERATURE = 0.8
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-distilroberta-v1"
ITEM_KEYWORD_EMBEDDING = "item_vector"
TOPK = 5
NUMBER_PRODUCTS = 1000
MAX_TEXT_LENGTH = 512
TEXT_EMBEDDING_DIMENSION = 768
DATA_PATH = "product_data.csv"

TEMPLATE_1 = "Create comma separated product keywords to perform a query on amazon dataset for this user input: {product_description}"
TEMPLATE_2 = """You are a salesman.Present the given product results in a nice way as answer to the user_msg. Don't ask questions back,
if results are empty just say that we don't have such products,
{chat_history}
user:{user_msg}
Chatbot:"""
