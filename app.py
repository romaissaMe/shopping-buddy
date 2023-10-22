import streamlit as st
from sentence_transformers import SentenceTransformer
from redis.commands.search.query import Query
import redis
from langchain.prompts import PromptTemplate
from langchain import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
import os
from dotenv import load_dotenv
import numpy as np

load_dotenv()
redis_key = os.getenv('REDIS_KEY')
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
repo_id = 'tiiuae/falcon-7b-instruct'

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text="", display_method='markdown'):
        self.container = container
        self.text = initial_text
        self.display_method = display_method

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token + " "
        display_function = getattr(self.container, self.display_method, None)
        if display_function is not None:
            display_function(self.text)
        else:
            raise ValueError(f"Invalid display_method: {self.display_method}")
        

st.title('My Amazon shopping buddy 🏷️')
st.caption('🤖 Powered by Falcon Open Source AI model')

#connect to redis database
@st.cache_resource()
def redis_connect():
  redis_conn = redis.Redis(
    host='redis-12882.c259.us-central1-2.gce.cloud.redislabs.com',
    port=12882,
    password=redis_key)
  return redis_conn

redis_conn = redis_connect()

#the encoding keywords chain
@st.cache_resource()
def encode_keywords_chain():
    falcon_llm_1 = HuggingFaceHub(repo_id = repo_id, model_kwargs={'temperature':0.1,'max_new_tokens':500},huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN)
    prompt = PromptTemplate(
        input_variables=["product_description"],
        template="Create comma seperated product keywords to perform a query on a amazon dataset for this user input: {product_description}",
    )
    chain = LLMChain(llm=falcon_llm_1, prompt=prompt)
    return chain
chain = encode_keywords_chain()
#the present products chain

@st.cache_resource()
def present_products_chain():
    template = """You are a salesman. Be kind, detailed and nice.  take the given context and Present the given queried search result in a nice way as answer to the user_msg. dont ask questions back or freestyle and invent followup conversation! 
    {chat_history}
    user:{user_msg}
    Chatbot:"""
    prompt = PromptTemplate(
        input_variables=["chat_history", "user_msg"],
        template=template
    )
    memory = ConversationBufferMemory(memory_key="chat_history")
    llm_chain = LLMChain(
        llm = ChatOpenAI(openai_api_key=os.getenv('OPENAI_API_KEY'),temperature=0.8,model='gpt-3.5-turbo'),
        prompt=prompt,
        verbose=False,
        memory=memory,
    )
    return llm_chain
 


llm_chain = present_products_chain()

@st.cache_resource()
def embedding_model():
    embedding_model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
    return embedding_model

embedding_model = embedding_model()

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hey im your online shopping buddy, how can i help you today?"}]
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

prompt = st.chat_input(key="user_input" )

if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message('user').write(prompt)
    st.session_state.disabled = True
    keywords = chain.run(prompt)
    #vectorize the query
    query_vector = embedding_model.encode(keywords)
    query_vector = np.array(query_vector).astype(np.float32).tobytes()
    #prepare the query
    ITEM_KEYWORD_EMBEDDING_FIELD = 'item_vector'
    topK=5
    q = Query(f'*=>[KNN {topK} @{ITEM_KEYWORD_EMBEDDING_FIELD} $vec_param AS vector_score]').sort_by('vector_score').paging(0,topK).return_fields('vector_score','item_name','item_id','item_keywords').dialect(2)
    params_dict = {"vec_param": query_vector}
    #Execute the query
    results = redis_conn.ft().search(q, query_params = params_dict)

    full_result_string = ''
    for product in results.docs:
        full_result_string += product.item_name + ' ' + product.item_keywords  + "\n\n\n"

    result = llm_chain.predict(user_msg=f"{full_result_string} ---\n\n {prompt}")
    st.session_state.messages.append({"role": "assistant", "content": result})
    st.chat_message('assistant').write(result)



