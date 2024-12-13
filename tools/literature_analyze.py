"""文献分析工具模块

本模块提供了使用LLM进行文献内容分析的功能。
主要用于分析文献中的关键发现、效果类型和证据强度。

依赖:
    - openai: OpenAI API接口
    - llama_index_core: 用于文档索引和查询
"""

import logging
from typing import Dict, Any
import json
from llama_index.core import Settings
from models.pubmed_models import PubMedLiterature

logger = logging.getLogger(__name__)

class LiteratureAnalyzer:
    """文献分析器类

    使用LLM分析医学文献内容，提取关键信息和结论。
    """

    def __init__(self):
        """初始化文献分析器"""

    def analyze_article(self, compound: str, disease: str, article: PubMedLiterature) -> Dict[str, Any]:
        """分析单篇文献的内容

        Args:
            compound (str): 化合物名称
            disease (str): 疾病名称
            article (PubMedLiterature): 要分析的文献

        Returns:
            Dict[str, Any]: 分析结果，包含效果类型、证据强度和关键发现
        """
        # 使用全局LLM配置
        llm = Settings.llm
        if llm is None:
            msg = "未配置全局LLM实例"
            logger.error(msg)
            raise RuntimeError(msg)

        # 构建分析提示
        prompt = f"""
        请分析以下医学文献的内容，查找关于{compound}与{disease}之间的治疗效果的关键发现，并提取以下关键信息：

        标题：{article.title}
        摘要：{article.abstract}

        请以JSON格式返回以下关键信息：
        {{
            "effect_type": "positive/negative/neutral",
            "evidence_strength": "strong/moderate/weak",
            "findings": ["发现1", "发现2", ...],
            "conclusion": "总结性结论"
        }}
        """

        logger.info("开始分析文献: PMID=%s", article.pmid)
        logger.debug("分析提示: %s", prompt)

        try:
            # 使用LLM进行分析
            response = llm.complete(prompt)
            response_text = response.text.strip()
            logger.info("获得LLM响应: %s", response_text[:200])  # 增加显示长度以便调试

            # 尝试提取JSON部分
            try:
                # 查找JSON开始和结束的位置
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = response_text[start:end]
                    result = json.loads(json_str)
                    logger.info("成功解析JSON响应")
                    return result
                else:
                    logger.error("响应中未找到有效的JSON")
                    raise ValueError("响应中未找到有效的JSON")
            except json.JSONDecodeError as e:
                logger.error("解析JSON响应失败: %s", str(e))
                raise
            except Exception as e:
                logger.error("处理响应失败: %s", str(e))
                raise

        except Exception as e:
            logger.error("分析文献失败: %s", str(e), exc_info=True)
            raise
