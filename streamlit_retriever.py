import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


import os
import pickle
import faiss
import numpy as np
import pandas as pd
from typing import List, Tuple
from tqdm import tqdm
#from sklearn.preprocessing import normalize
from text2vec import SentenceModel

import streamlit as st
import pandas as pd
from typing import List, Tuple
import numpy as np
from text2vec import SentenceModel
import faiss
import os

from src.ExcelIndexer import ExcelIndexer

@st.cache_resource  # 缓存模型
def get_model(model_path:str = 'shibing624/text2vec-base-chinese'):
    model = SentenceModel(model_path)
    return model

@st.cache_resource # 缓存检索器
def create_retriever(vector_sz:int, dataset_path:str, id_column:str, _model):  #_model: SentenceModel，加_是为了能够缓存
    retriever = ExcelIndexer(vector_sz=vector_sz, model = _model, embeddings_file='term_work/embeddings-10k.npy')
    retriever.load_excel(dataset_path,id_column)
    return retriever


# 在创建 retriever 时直接传入缓存的模型
model = get_model()
retriever = create_retriever(768, 'term_work/sampled-10k.xlsx', '描述', _model=model)


# Streamlit app
st.title("Excel Data Retrieval Visualization")
st.write("Upload an Excel file and enter a query to retrieve similar entries.")


id_column = "描述"


# Query input
query = st.text_input("Enter a search query:")
top_k = st.slider("Select number of results to display", min_value=1, max_value=100, value=5)

# Search and display results
if st.button("Search") and query:
    texts, scores = retriever.search_return_text(query, top_k)[0]  # 获取搜索结果

    st.write("### Results:")

    # 使用 expander 来折叠显示结果
    with st.expander("检索结果列表(点击展开)"):
        for j, text in enumerate(texts):
            st.markdown(
                f"""
                <div style="border:1px solid #ccc; padding:10px; border-radius:5px; margin-bottom:10px; background-color:#f9f9f9;">
                    <p><b>Text {j+1}:</b> {text}</p>
                    <p><b>Score:</b> {scores[j]:.4f}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
