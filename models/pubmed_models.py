"""PubMed文献数据模型模块

本模块定义了用于表示PubMed文献信息的数据模型。
使用Pydantic库实现数据验证和序列化功能。

主要模型：
    - PubMedLiterature: 单篇PubMed文献的数据模型
    - AnalysisReport: 文献分析报告的数据模型

用途：
    - 规范化文献数据的存储格式
    - 提供数据验证和类型检查
    - 支持JSON序列化和反序列化
"""

from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel

class PubMedLiterature(BaseModel):
    """PubMed文献数据模型

    用于存储从PubMed获取的单篇文献信息，包括文献的基本信息和元数据。

    Attributes:
        pmid (str): PubMed文献唯一标识符
        title (str): 文献标题
        abstract (str): 文献摘要
        authors (List[str]): 作者列表
        publication_date (datetime): 发表日期
        journal (str): 期刊名称
        keywords (List[str]): 关键词列表
        conclusion (Optional[str]): 结论部分（如果有）
        effect_type (Optional[str]): 效果类型（正面、负面或中性）
        evidence_strength (Optional[str]): 证据强度（强、中、弱）
    """
    pmid: str
    title: str
    abstract: str
    authors: List[str]
    publication_date: datetime
    journal: str
    keywords: List[str] = []  # 默认为空列表
    conclusion: Optional[str] = None  # 可选字段，默认为None
    effect_type: Optional[str] = None  # 可选字段，用于标记效果类型
    evidence_strength: Optional[str] = None  # 可选字段，用于标记证据强度


class AnalysisReport(BaseModel):
    """文献分析报告数据模型

    用于存储对特定化合物和疾病组合的文献分析结果。
    整合了多篇文献的统计信息和关键发现。

    Attributes:
        compound (str): 分析的化合物名称
        disease (str): 分析的疾病名称
        total_papers (int): 分析的文献总数
        positive_effects (int): 报告正面效果的文献数量
        negative_effects (int): 报告负面效果的文献数量
        neutral_effects (int): 报告中性效果的文献数量
        key_findings (List[str]): 关键发现列表
        evidence_summary (str): 证据总结
        analyzed_papers (List[PubMedLiterature]): 已分析的文献列表
        analysis_date (datetime): 分析完成的时间
    """
    compound: str
    disease: str
    total_papers: int
    positive_effects: int
    negative_effects: int
    neutral_effects: int
    key_findings: List[str]
    evidence_summary: str
    analyzed_papers: List[PubMedLiterature]
    analysis_date: datetime
