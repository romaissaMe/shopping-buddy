from langchain.prompts import PromptTemplate
from langchain import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from redis.commands.search.query import Query
import time
import os
from dotenv import load_dotenv
import numpy as np
load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
repo_id = 'tiiuae/falcon-7b-instruct'

falcon_llm_1 = HuggingFaceHub(repo_id = repo_id, model_kwargs={'temperature':0.1,'max_new_tokens':500},huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN)

prompt = PromptTemplate(
    input_variables=["product_description"],
    template="Create comma seperated product keywords to perform a query on a amazon dataset for this user input: {product_description}",
)

chain = LLMChain(llm=falcon_llm_1, prompt=prompt)

# code The response
repo_id_2 = 'tiiuae/falcon-7b' 
template = """You are a salesman. Be kind, detailed and nice.  take the given context and Present the given queried search result in a nice way as answer to the user_msg. dont ask questions back or freestyle and invent followup conversation! just 

{chat_history}
{user_msg}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "user_msg"],
    template=template
)
memory = ConversationBufferMemory(memory_key="chat_history")

llm_chain = LLMChain(
    llm = HuggingFaceHub(repo_id = repo_id_2, model_kwargs={'temperature':0.8,'max_new_tokens':500}),
    prompt=prompt,
    verbose=False,
    memory=memory,
)





