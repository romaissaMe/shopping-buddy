from langchain.embeddings import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from database import redis_conn
from utilities import create_flat_index, load_vectors



#set maximum length for text fields
MAX_TEXT_LENGTH = 512

def auto_truncate(text:str):
    return text[0:MAX_TEXT_LENGTH]

data = pd.read_csv('product_data.csv',converters={'bullet_point':auto_truncate,'item_keywords':auto_truncate,'item_name':auto_truncate})
data['primary_key'] = data['item_id'] + '-' + data['domain_name']
data.drop(columns=['item_id','domain_name'],inplace=True)
data['item_keywords'].replace('',np.nan,inplace=True)
data.dropna(subset=['item_keywords'],inplace=True)
data.reset_index(drop=True, inplace=True)
data_metadata = data.head(500).to_dict(orient='index')

#generating embeddings (vectors) for the item keywords
embedding_model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
# embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)

#get the item keywords attribute for each product and encode them into vector embeddings
item_keywords = [data_metadata[i]['item_keywords'] for i in data_metadata.keys()]
item_keywords_vectors = [embedding_model.encode(item) for item in item_keywords]

TEXT_EMBEDDING_DIMENSION=768
NUMBER_PRODUCTS=500

print ('Loading and Indexing + ' +  str(NUMBER_PRODUCTS) + ' products')
#flush all data
redis_conn.flushall()
#create flat index & load vectors
create_flat_index(redis_conn,NUMBER_PRODUCTS,TEXT_EMBEDDING_DIMENSION,'COSINE')
load_vectors(redis_conn,data_metadata,item_keywords_vectors)






