"""向量存储模块

该模块提供了基于PostgreSQL的向量数据库存储和检索功能。主要用于存储和管理化合物-疾病分析的向量表示及其相关信息。

主要组件:
    - PubMedMemory: 数据库模型类，用于存储PubMed文献
    - AnalysisMemory: 数据库模型类，用于存储分析记录
    - AnalysisVectorStore: 向量数据库管理类，提供存储和检索功能
    - PubMedVectorStore: PubMed文献的向量存储管理类

技术特点:
    - 使用pgvector扩展实现向量相似度搜索
    - 基于SQLAlchemy ORM进行数据库操作
    - 支持余弦距离的相似度计算
    - 支持混合搜索（关键词 + 语义）
"""
import os
from datetime import datetime
import logging
from typing import List, Optional, Tuple
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Integer, String, DateTime, create_engine, text
from sqlalchemy.orm import Session, declarative_base
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.postgres import PGVectorStore as LlamaIndexPGStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from models.pubmed_models import PubMedLiterature

logger = logging.getLogger(__name__)

# 创建基类
Base = declarative_base()

class PubMedMemory(Base):
    """PubMed文献存储模型类

    用于存储PubMed文献的数据库模型。

    属性:
        id (int): 主键ID
        pmid (str): PubMed文献ID
        title (str): 文献标题
        abstract (str): 文献摘要
        content_vector (Vector): 文本内容的向量表示
        compound (str): 相关化合物名称
        disease (str): 相关疾病名称
        created_at (DateTime): 创建时间
    """
    __tablename__ = 'pubmed_memory'

    id = Column(Integer, primary_key=True)
    pmid = Column(String, unique=True)
    title = Column(String)
    abstract = Column(String)
    content_vector = Column(Vector(1536))  # 使用numpy数组
    compound = Column(String)
    disease = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<PubMedMemory(pmid='{self.pmid}', title='{self.title[:30]}...')>"


class AnalysisMemory(Base):
    """分析记忆模型类

    用于存储化合物和疾病分析的数据库模型。

    属性:
        id (int): 主键ID
        compound (str): 化合物名称
        disease (str): 疾病名称
        content_vector (Vector): 内容的向量表示
        content (str): 分析内容
        created_at (DateTime): 创建时间
    """
    __tablename__ = 'analysis_memory'

    id = Column(Integer, primary_key=True)
    compound = Column(String)
    disease = Column(String)
    content_vector = Column(Vector(1536))  # 使用numpy数组
    content = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<AnalysisMemory(compound='{self.compound}', disease='{self.disease}')>"


class AnalysisVectorStore:
    """向量存储类

    用于管理和检索分析记忆的向量数据库接口。

    属性:
        engine: SQLAlchemy数据库引擎实例
        embed_model: OpenAI Embedding模型实例
    """

    def __init__(self, connection_string: str, embed_model: Optional[OpenAIEmbedding] = None):
        """初始化向量存储

        Args:
            connection_string (str): 数据库连接字符串
            embed_model (Optional[OpenAIEmbedding]): OpenAI Embedding模型实例
        """
        self.engine = create_engine(connection_string)
        Base.metadata.create_all(self.engine)
        self.embed_model = embed_model or OpenAIEmbedding(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY")
        )

    def store_analysis(self, compound: str, disease: str, content: str):
        """存储分析结果及其向量嵌入

        Args:
            compound (str): 化合物名称
            disease (str): 疾病名称
            content (str): 分析内容
        """
        try:
            # 生成内容的向量表示
            vector = self.embed_model.get_text_embedding(content)

            # 创建新的分析记录
            analysis = AnalysisMemory(
                compound=compound,
                disease=disease,
                content=content,
                content_vector=vector,  # 直接使用numpy数组
                created_at=datetime.utcnow()
            )

            # 保存到数据库
            with Session(self.engine) as session:
                session.add(analysis)
                session.commit()
                logger.info("成功存储分析结果")

        except Exception as e:
            # 简化错误信息，只记录关键信息
            error_msg = f"存储分析结果失败 (compound={compound}, disease={disease}): {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def search_similar(self, query_text: str, limit: int = 5) -> List[Tuple[AnalysisMemory, float]]:
        """使用向量相似度搜索相似的分析记录

        Args:
            query_text (str): 查询文本
            limit (int): 返回结果的最大数量

        Returns:
            List[Tuple[AnalysisMemory, float]]: 相似度排序后的分析记录和得分列表
        """
        try:
            # 生成查询向量
            query_vector = self.embed_model.get_text_embedding(query_text)

            with Session(self.engine) as session:
                results = session.query(
                    AnalysisMemory,
                    (1 - AnalysisMemory.content_vector.cosine_distance(query_vector)
                     ).label('similarity')
                )\
                    .order_by(text('similarity DESC'))\
                    .limit(limit)\
                    .all()

                return [(item[0], float(item[1])) for item in results]
        except Exception as e:
            logger.error("搜索相似分析失败: %s", str(e), exc_info=True)
            raise

    def get_analysis_history(self, compound: str, disease: str) -> List[AnalysisMemory]:
        """获取指定化合物和疾病的历史分析记录

        Args:
            compound (str): 化合物名称
            disease (str): 疾病名称

        Returns:
            List[AnalysisMemory]: 按时间倒序排列的分析记录列表
        """
        with Session(self.engine) as session:
            return session.query(AnalysisMemory)\
                .filter_by(compound=compound, disease=disease)\
                .order_by(AnalysisMemory.created_at.desc())\
                .all()


class PubMedVectorStore:
    """PubMed文献向量存储类

    用于管理和检索PubMed文献的向量数据库接口。
    支持混合搜索（关键词 + 语义相似度）。

    属性:
        engine: SQLAlchemy数据库引擎实例
        embed_model: OpenAI Embedding模型实例
        vector_store: LlamaIndex的PG向量存储实例
    """

    def __init__(self, connection_string: str, embed_model: Optional[OpenAIEmbedding] = None):
        """初始化PubMed向量存储

        Args:
            connection_string (str): 数据库连接字符串
            embed_model (Optional[OpenAIEmbedding]): OpenAI Embedding模型实例
        """
        self.engine = create_engine(connection_string)
        Base.metadata.create_all(self.engine)
        self.embed_model = embed_model or OpenAIEmbedding(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.connection_string = connection_string

    def store_article(self, article: PubMedLiterature, compound: str, disease: str):
        """存储PubMed文献及其向量表示

        Args:
            article: PubMed文献对象
            compound (str): 化合物名称
            disease (str): 疾病名称
        """
        try:
            # 生成文本内容的向量表示
            content = f"{article.title}\n{article.abstract}"
            vector = self.embed_model.get_text_embedding(content)

            # 创建新的文献记录
            memory = PubMedMemory(
                pmid=article.pmid,
                title=article.title,
                abstract=article.abstract,
                content_vector=vector,  # 直接使用numpy数组
                compound=compound,
                disease=disease,
                created_at=datetime.utcnow()
            )

            # 保存到数据库
            with Session(self.engine) as session:
                session.add(memory)
                session.commit()
                logger.info("成功存储文献: PMID=%s", article.pmid)

        except Exception as e:
            # 检查是否是唯一键冲突
            if "duplicate key value violates unique constraint" in str(e):
                error_msg = f"文献已存在 (PMID={article.pmid})"
                logger.warning(error_msg)
                raise RuntimeError(error_msg) from None  # 使用 from None 简化堆栈跟踪
            else:
                # 其他错误
                error_msg = f"存储文献失败 (PMID={article.pmid}): {str(e)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e

    def hybrid_search(self, query: str, limit: int = 5) -> List[Tuple[PubMedMemory, float]]:
        """执行混合搜索（关键词 + 语义相似度）

        使用LlamaIndex的向量存储进行混合搜索，结合了：
        1. 语义相似度搜索（通过向量相似度）
        2. 关键词匹配（通过元数据过滤）
        3. BM25文本相关度（通过PostgreSQL的全文搜索）

        Args:
            query (str): 搜索查询
            compound (Optional[str]): 过滤特定化合物
            disease (Optional[str]): 过滤特定疾病
            limit (int): 返回结果的最大数量

        Returns:
            List[Tuple[PubMedMemory, float]]: 搜索结果和相似度得分
        """
        try:
            # 从数据库建索引
            vector_store = LlamaIndexPGStore.from_params(
                connection_string=self.connection_string,
                table_name="pubmed_memory",
                embed_dim=1536,
                hybrid_search=True,

            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                storage_context=storage_context,
                show_progress=True
            )

            # 如果索引为空，返回空结果
            if not index:
                return []

            # 使用VectorStoreIndex执行混合搜索
            retriever = index.as_retriever(
                similarity_top_k=limit
            )
            nodes_with_scores = retriever.retrieve(query)

            # 直接从节点中获取信息并构建结果
            results = [
                (PubMedMemory(
                    pmid=node.metadata["pmid"],
                    title=node.metadata["title"],
                    abstract=node.metadata["abstract"],
                    compound=node.metadata["compound"],
                    disease=node.metadata["disease"]
                ), score)
                for node, score in nodes_with_scores
            ]

            return results

        except Exception as e:
            logger.error("混合搜索失败: %s", str(e), exc_info=True)
            raise
