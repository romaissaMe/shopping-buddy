from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from redis.commands.search.query import Query
import time
import os
from dotenv import load_dotenv
import numpy as np
from database import redis_conn

load_dotenv()

llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.3, openai_api_key=os.getenv('OPENAI_API_KEY'))
prompt = PromptTemplate(
    input_variables=["product_description"],
    template="Create comma seperated product keywords to perform a query on a amazon dataset for this user input: {product_description}",
)

chain = LLMChain(llm=llm, prompt=prompt)

userinput = input("Hey im a E-commerce Chatbot, how can i help you today? ")
print("User:", userinput)
# Run the chain only specifying the input variable.
keywords = chain.run(userinput)

embedding_model = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))
#vectorize the query
query_vector = embedding_model.embed_query(keywords)
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
    full_result_string += product.item_name + ' ' + product.item_keywords + ' ' + product.item_id + "\n\n\n"

# code The response
template = """You are a chatbot. Be kind, detailed and nice. Present the given queried search result in a nice way as answer to the user input. dont ask questions back! just take the given context

{chat_history}
Human: {user_msg}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "user_msg"],
    template=template
)
memory = ConversationBufferMemory(memory_key="chat_history")
llm_chain = LLMChain(
    llm=OpenAI(model_name="gpt-3.5-turbo", temperature=0.8, openai_api_key=os.getenv('OPENAI_API_KEY')),
    prompt=prompt,
    verbose=False,
    memory=memory,
)

answer = llm_chain.predict(user_msg=f"{full_result_string} ---\n\n {userinput}")
print("Bot:", answer)
time.sleep(0.5)

while True:
    follow_up = input("Anything else you want to ask about this topic?")
    print("User:", follow_up)
    answer = llm_chain.predict(
        user_msg=follow_up
    )
    print("Bot:", answer)
    time.sleep(0.5)
