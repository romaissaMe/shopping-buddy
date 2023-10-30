FALCON_REPO_ID = "tiiuae/falcon-7b-instruct"
FALCON_TEMPERATURE = 0.1
FALCON_MAX_TOKENS = 500

OPENAI_MODEL_NAME = "gpt-3.5-turbo"
OPENAI_TEMPERATURE = 0.8

EMBEDDING_MODEL_NAME = "sentence-transformers/all-distilroberta-v1"

TEMPLATE_1 = "Create comma seperated product keywords to perform a query on a amazon dataset for this user input: {product_description}"
TEMPLATE_2 = """You are a salesman.Present the given product results in a nice way as answer to the user_msg. Dont ask questions back,
    {chat_history}
    user:{user_msg}
    Chatbot:"""
