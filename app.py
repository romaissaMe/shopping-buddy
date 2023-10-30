import os

import numpy as np
import redis
import streamlit as st
from dotenv import load_dotenv
from langchain import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from redis.commands.search.query import Query
from sentence_transformers import SentenceTransformer

from constants import (
    EMBEDDING_MODEL_NAME,
    FALCON_MAX_TOKENS,
    FALCON_REPO_ID,
    FALCON_TEMPERATURE,
    OPENAI_MODEL_NAME,
    OPENAI_TEMPERATURE,
    TEMPLATE_1,
    TEMPLATE_2,
)
from database import create_redis

load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
ITEM_KEYWORD_EMBEDDING = "item_vector"
TOPK = 5


def main():
    # connect to redis database
    @st.cache_resource()
    def connect_to_redis():
        pool = create_redis()
        return redis.Redis(connection_pool=pool)

    # the encoding keywords chain
    @st.cache_resource()
    def encode_keywords_chain():
        falcon_llm_1 = HuggingFaceHub(
            repo_id=FALCON_REPO_ID,
            model_kwargs={"temperature": FALCON_TEMPERATURE, "max_new_tokens": FALCON_MAX_TOKENS},
            huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
        )
        prompt = PromptTemplate(
            input_variables=["product_description"],
            template=TEMPLATE_1,
        )
        chain = LLMChain(llm=falcon_llm_1, prompt=prompt)
        return chain

    # the present products chain
    @st.cache_resource()
    def present_products_chain():
        template = TEMPLATE_2
        prompt = PromptTemplate(input_variables=["chat_history", "user_msg"], template=template)
        memory = ConversationBufferMemory(memory_key="chat_history")
        llm_chain = LLMChain(
            llm=ChatOpenAI(
                openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=OPENAI_TEMPERATURE, model=OPENAI_MODEL_NAME
            ),
            prompt=prompt,
            verbose=False,
            memory=memory,
        )
        return llm_chain

    @st.cache_resource()
    def instance_embedding_model():
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return embedding_model

    st.title("My Amazon shopping buddy 🏷️")
    st.caption("🤖 Powered by Falcon Open Source AI model")
    redis_conn = connect_to_redis()
    keywords_chain = encode_keywords_chain()
    chat_chain = present_products_chain()
    embedding_model = instance_embedding_model()

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hey im your online shopping buddy, how can i help you today?"}
        ]
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

    prompt = st.chat_input(key="user_input")

    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        st.session_state.disabled = True
        keywords = keywords_chain.run(prompt)
        # vectorize the query
        query_vector = embedding_model.encode(keywords)
        query_vector_bytes = np.array(query_vector).astype(np.float32).tobytes()
        # prepare the query

        q = (
            Query(f"*=>[KNN {TOPK} @{ITEM_KEYWORD_EMBEDDING} $vec_param AS vector_score]")
            .sort_by("vector_score")
            .paging(0, TOPK)
            .return_fields("vector_score", "item_name", "item_id", "item_keywords")
            .dialect(2)
        )
        params_dict = {"vec_param": query_vector_bytes}
        # Execute the query
        results = redis_conn.ft().search(q, query_params=params_dict)
        result_output = ""
        for product in results.docs:
            result_output += f"product_name:{product.item_name}, product_description:{product.item_keywords} \n"
        result = chat_chain.predict(user_msg=f"{result_output}\n{prompt}")
        st.session_state.messages.append({"role": "assistant", "content": result})
        st.chat_message("assistant").write(result)


if __name__ == "__main__":
    main()
