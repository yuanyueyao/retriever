# Sim-Retriever

You can easily use this to build a vector-based document retriever, and the documents should be stored in an xlsx format. Later, support for other formats can also be added.

## quick start

```powershell
cd retriever
streamlit run streamlit_retriever.py model_path=path/to/model vector_size=768 dataset_path=path/to/dataset retrieve_column = 'retrieve_column_name' 
```

## environment

`pip install -r requirements.txt`
