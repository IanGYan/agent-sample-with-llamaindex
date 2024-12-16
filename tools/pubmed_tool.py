"""PubMed文献检索工具模块

本模块提供了与 PubMed 数据库交互的功能，用于检索和获取医学文献信息。
主要功能包括：
1. 按照化合物和疾病名称搜索相关文献
2. 支持年份范围过滤
3. 获取文献的详细信息（标题、摘要、作者等）
4. 解析和标准化PubMed日期格式

依赖:
    - Bio.Entrez: PubMed API接口
    - Bio.Medline: MEDLINE格式解析
"""

from typing import List, Optional
import datetime
from Bio import Entrez, Medline
from models.pubmed_models import PubMedLiterature
import logging

logger = logging.getLogger(__name__)

class PubMedTool:
    """PubMed文献检索工具类

    该类封装了与PubMed数据库交互的主要功能，提供了便捷的文献检索和数据获取方法。

    主要功能:
        - 初始化PubMed API访问凭证
        - 搜索特定化合物和疾病相关的文献
        - 支持按年份范围过滤搜索结果
        - 获取文献的详细信息
        - 解析PubMed日期格式

    Attributes:
        email (str): PubMed API访问所需的邮箱地址
        api_key (str): PubMed API访问密钥
    """

    def __init__(self, email: str, api_key: str):
        """初始化PubMed工具

        Args:
            email (str): PubMed API访问所需的邮箱
            api_key (str): PubMed API密钥
        """
        Entrez.email = email
        Entrez.api_key = api_key

    def search_pubmed(self, compound: str, disease: str, max_results: int = 5,
                      start_year: Optional[int] = None, end_year: Optional[int] = None) -> List[str]:
        """在指定年份范围内搜索化合物对疾病的治疗效果相关文献

        Args:
            compound (str): 要搜索的化合物名称
            disease (str): 要搜索的疾病名称
            max_results (int, optional): 最大返回结果数量，默认为5
            start_year (Optional[int], optional): 起始年份，默认为None
            end_year (Optional[int], optional): 结束年份，默认为None

        Returns:
            List[str]: PubMed文献ID列表
        """
        query = f"{compound} AND {disease} AND (treatment OR therapy OR effect)"

        # 添加日期范围过滤（如果指定）
        if start_year and end_year:
            query += f" AND ({start_year}[PDAT]:{end_year}[PDAT])"
        elif start_year:
            query += f" AND {start_year}[PDAT]:3000[PDAT]"
        elif end_year:
            query += f" AND 1900[PDAT]:{end_year}[PDAT]"

        try:
            handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
            records = Entrez.read(handle)
            handle.close()

            return [str(id_) for id_ in records["IdList"]]

        except Exception as e:
            logger.error("PubMed搜索失败: %s", str(e), exc_info=True)
            return []

    def fetch_articles(self, id_list: List[str]) -> List[PubMedLiterature]:
        """获取指定PubMed ID列表的文献详细信息

        Args:
            id_list (List[str]): PubMed文献ID列表

        Returns:
            List[PubMedLiterature]: 文献详细信息列表
        """
        if not id_list:
            return []

        try:
            handle = Entrez.efetch(db="pubmed", id=id_list,
                                rettype="medline", retmode="text")
            records = list(Medline.parse(handle))
            articles = []

            for record in records:
                pmid = record.get('PMID')
                if not pmid:
                    continue

                title = record.get('TI', '')
                abstract = record.get('AB', '')
                date_str = record.get('DP', '')
                authors = record.get('AU', [])  # 作者通常是列表
                journal = record.get('TA', '')  # 期刊缩写名称
                keywords = record.get('MH', [])  # MeSH关键词列表

                article = PubMedLiterature(
                    pmid=pmid,
                    title=title,
                    abstract=abstract,
                    publication_date=self._parse_date(date_str),
                    authors=authors,
                    journal=journal,
                    keywords=keywords
                )
                articles.append(article)

            handle.close()
            return articles
        except Exception as e:
            logger.error("获取PubMed文献详情失败: %s", str(e), exc_info=True)
            return []

    def _parse_date(self, date_str: str) -> datetime.datetime:
        """解析PubMed日期字符串为datetime对象

        Args:
            date_str (str): PubMed日期字符串

        Returns:
            datetime.datetime: 解析后的日期对象，解析失败则返回当前时间
        """
        try:
            # 处理各种日期格式
            if ' ' in date_str:
                date_str = date_str.split(' ')[0]
            if len(date_str) >= 4:
                year = int(date_str[:4])
                month = int(date_str[5:7]) if len(date_str) >= 7 else 1
                day = int(date_str[8:10]) if len(date_str) >= 10 else 1
                return datetime.datetime(year, month, day)
        except (ValueError, IndexError):
            return datetime.datetime.now()
        return datetime.datetime.now()
