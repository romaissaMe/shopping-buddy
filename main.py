import streamlit as st
from chatbot import llm_chain, chain
from sentence_transformers import SentenceTransformer
from redis.commands.search.query import Query
from database import redis_conn
import numpy as np



st.title('My Amazon shopping buddy 🏷️')
st.caption('🤖 Powered by Falcon Open Source AI model')
st.session_state['disabled']= False

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hey im your online shopping buddy, how can i help you today?"}]
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

prompt = st.chat_input(key="user_input",disabled=st.session_state.disabled )
embedding_model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
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



