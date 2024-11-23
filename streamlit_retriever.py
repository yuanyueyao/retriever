import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import os
import pickle
import faiss
import numpy as np
import pandas as pd
from typing import List, Tuple
from tqdm import tqdm
from text2vec import SentenceModel
import streamlit as st
import sys
from src.ExcelIndexer import ExcelIndexer

# 命令行参数处理函数
def get_cli_args():
    args = {}
    # 跳过第一个参数（脚本名）和第二个参数（streamlit run）
    argv = sys.argv[2:] if len(sys.argv) > 2 else []
    
    for arg in argv:
        if '=' in arg:
            key, value = arg.split('=', 1)
            args[key.strip()] = value.strip()
    return args

# 获取命令行参数
cli_args = get_cli_args()

# 设置默认值，可以被命令行参数覆盖
DEFAULT_CONFIG = {
    'model_path': 'shibing624/text2vec-base-chinese',
    'dataset_path': '../term_work/sampled-10k.xlsx',
    'embeddings_file': '../term_work/embeddings-10k.npy',
    'vector_size': 768,
    'id_column': '描述'
}

# 合并默认配置和命令行参数
config = DEFAULT_CONFIG.copy()
config.update(cli_args)

# 将vector_size转换为整数
config['vector_size'] = int(config['vector_size'])

@st.cache_resource
def get_model(model_path: str = config['model_path']):
    model = SentenceModel(model_path)
    return model

@st.cache_resource
def create_retriever(vector_sz: int, dataset_path: str, id_column: str, _model, embeddings_file: str):
    retriever = ExcelIndexer(vector_sz=vector_sz, model=_model, embeddings_file=embeddings_file)
    retriever.load_excel(dataset_path, id_column)
    return retriever

# 在页面显示当前配置
if st.sidebar.checkbox("Show Configuration"):
    st.sidebar.write("Current Configuration:")
    for key, value in config.items():
        st.sidebar.write(f"{key}: {value}")

# 初始化模型和检索器
model = get_model(config['model_path'])
retriever = create_retriever(
    config['vector_size'],
    config['dataset_path'],
    config['id_column'],
    _model=model,
    embeddings_file=config['embeddings_file']
)

# Streamlit app
st.title("Excel Data Retrieval Visualization")
st.write("Upload an Excel file and enter a query to retrieve similar entries.")

# Query input
query = st.text_input("Enter a search query:")
top_k = st.slider("Select number of results to display", min_value=1, max_value=100, value=5)

# Search and display results
if st.button("Search") and query:
    texts, scores = retriever.search_return_text(query, top_k)[0]

    st.write("### Results:")

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