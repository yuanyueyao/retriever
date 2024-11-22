import os
from typing import List, Tuple

import faiss
import numpy as np
import pandas as pd

from tqdm import tqdm
#from sklearn.preprocessing import normalize
from text2vec import SentenceModel

os.environ['CUDA_VISIBLE_DEVICES'] = '3'


class ExcelIndexer(object):
    '''
        excel表格检索器
    '''
    def __init__(self, vector_sz: int, n_subquantizers=0, n_bits=8, model: SentenceModel = None, embeddings_file: str = None):
        """初始化索引器，选择使用FAISS的类型"""
        if n_subquantizers > 0:
            self.index = faiss.IndexPQ(vector_sz, n_subquantizers, n_bits, faiss.METRIC_INNER_PRODUCT)
        else:
            self.index = faiss.IndexFlatIP(vector_sz)
        self.index_id_to_row_id = []  # 用于存储 FAISS 索引 ID 到 Excel 文件中的行号映射

        self.data_frame = None  # 存储 Excel 数据
        self.model = model
        self.embeddings_file = embeddings_file  # 存储嵌入向量文件路径

        print(f'Initialized FAISS index of type {type(self.index)}')

    def load_excel(self, excel_file: str, id_column: str, batch_size: int = 2048):
        """
        加载 Excel 文件并自动创建 FAISS 索引
        :param excel_file: Excel 文件路径
        :param vector_column: 存储嵌入向量的列名
        :param id_column: 存储唯一标识符的列名
        """
        # 加载 Excel 文件到 DataFrame
        print('Loading Excel file...')
        self.data_frame = pd.read_excel(excel_file)
        print(f'Loaded Excel file: {excel_file}, total rows: {len(self.data_frame)}')
        lenth = len(self.data_frame)

        # 如果存在保存的嵌入向量文件，直接加载它
        if self.embeddings_file and os.path.exists(self.embeddings_file):
            print(f'Loading embeddings from {self.embeddings_file}...')
            self.embeddings = np.load(self.embeddings_file)
            print(f'Loaded embeddings from file, shape: {self.embeddings.shape}')
            print('Indexing data...')
            ids = range(lenth)
            self.index_data(ids, self.embeddings)
            print(f'Total data indexed: {len(self.index_id_to_row_id)}')

        else:
            self.embeddings = np.empty((lenth, self.index.d), dtype=np.float32)  # 初始化一个空的嵌入向量矩阵
            # 提取 ID 和向量数据
            print('Indexing data...')
            for times in range(lenth // batch_size + 1):
                print(f'Indexing batch {times + 1}/{lenth // batch_size + 1}, total indexed: {len(self.index_id_to_row_id)}, total data: {lenth}')
                start = times * batch_size
                end = min((times + 1) * batch_size, lenth)
                ids = range(start, end)
                embeddings = np.array([self.model.encode(self.data_frame[id_column][i]) for i in range(start, end)]).astype('float32')

                # 保存嵌入向量到内存中
                self.embeddings[start:end] = embeddings
                self.index_data(ids, embeddings)

            # 保存嵌入向量到文件
            if self.embeddings_file:
                print(f'Saving embeddings to {self.embeddings_file}...')
                np.save(self.embeddings_file, self.embeddings)

        print('Indexing done!')

    def index_data(self, ids: List[int], embeddings: np.array):
        """
        将数据从 Excel 中加载并索引
        :param ids: 来自 Excel 的行 ID（可以是某一列唯一标识符）
        :param embeddings: 行的嵌入向量
        """
        # 更新 ID 映射
        self._update_id_mapping(ids)

        # 将 embeddings 转换为 float32 类型
        embeddings = embeddings.astype('float32')

        # 如果 FAISS 索引尚未训练，则进行训练
        if not self.index.is_trained:
            self.index.train(embeddings)

        # 将 embeddings 添加到 FAISS 索引
        self.index.add(embeddings)
        print(f'Total data indexed: {len(self.index_id_to_row_id)}')

    # 其他方法保持不变

    def _update_id_mapping(self, row_ids: List[int]):
        """更新行 ID 到索引 ID 的映射关系"""
        self.index_id_to_row_id.extend(row_ids)

    def search_return_text(self, query: str, top_docs: int, index_batch_size: int = 10000) -> List[Tuple[List[object], List[float]]]:
        search_result = self.search_dp(query, top_docs, index_batch_size)
        result = []
        for i in search_result:
            result.append(([self.data_frame['描述'][int(j)] for j in i[0] if len(self.data_frame['描述'][int(j)]) > 20 ],i[1]))  #过滤掉描述长度小于10的
        return result
    
    def search_dp(self, query: str, top_docs: int, index_batch_size: int = 2048) -> List[Tuple[List[object], List[float]]]:
        """
        执行 dp 查询，返回 Excel 文件行 ID 和相似度得分
        :param query_vectors: 查询的嵌入向量
        :param top_docs: 返回的最近邻文档数量
        :param index_batch_size: 每批次处理的查询数量
        :return: 返回每个查询向量对应的最近邻行 ID 和得分
        """
        query_vectors = self.model.encode([query]).astype('float32')
        result = []  # 存储所有查询结果

        # 计算批次数量
        nbatch = (len(query_vectors) - 1) // index_batch_size + 1

        # 批量处理查询
        for k in tqdm(range(nbatch)):
            start_idx = k * index_batch_size
            end_idx = min((k + 1) * index_batch_size, len(query_vectors))

            q = query_vectors[start_idx: end_idx]

            # 使用 FAISS 进行搜索
            scores, indexes = self.index.search(q, top_docs)

            # 将 FAISS 索引 ID 转换为 Excel 中的行 ID
            db_ids = [[str(self.index_id_to_row_id[i]) for i in query_top_idxs] for query_top_idxs in indexes]

            # 将每个查询结果添加到最终结果中
            result.extend([(db_ids[i], scores[i]) for i in range(len(db_ids))])

        return result