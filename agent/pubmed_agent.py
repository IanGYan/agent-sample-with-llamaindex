"""LlamaIndex Agent模块

本模块实现了基于LlamaIndex的ReAct智能代理，支持多轮对话和自动任务执行。
"""

import os
import logging
from typing import List, Dict, Any, Optional, cast
from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool, BaseTool
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.embeddings.openai import OpenAIEmbedding
from tools.pubmed_tool import PubMedTool
from tools.literature_analyze import LiteratureAnalyzer
from memory.vector_store import PubMedVectorStore, AnalysisVectorStore

logger = logging.getLogger(__name__)
log_level = os.getenv('LOG_LEVEL', 'ERROR')  # 如果未设置，默认使用 ERROR
numeric_level = getattr(logging, log_level.upper(), logging.ERROR)
logger.setLevel(numeric_level)  # 设置日志级别为从环境变量中读取的值

class MedLiteratureAnalysisAgent:
    """医学文献分析智能代理类

    使用ReAct Agent实现多轮对话和自动任务执行。
    整合PubMed检索和文献分析功能，提供自然语言交互接口。
    """

    def __init__(self, vector_db_url: str):
        """初始化代理

        Args:
            vector_db_url (str): 向量数据库连接字符串
        Raises:
            ValueError: 当缺少必要的环境变量时抛出
        """
        # 检查必要的环境变量
        required_env_vars = {
            "ENTREZ_EMAIL": "PubMed API 邮箱",
            "ENTREZ_API_KEY": "PubMed API 密钥",
            "OPENAI_API_KEY": "OpenAI API 密钥"
        }

        missing_vars = []
        env_values = {}
        for var, desc in required_env_vars.items():
            value = os.getenv(var)
            if not value or value.strip() == "":
                missing_vars.append(f"{desc} ({var})")
            else:
                env_values[var] = value.strip()

        if missing_vars:
            raise ValueError(f"缺少必要的环境变量或环境变量为空: {', '.join(missing_vars)}")

        try:
            # 设置调试处理器
            debug_handler = LlamaDebugHandler(print_trace_on_end=(log_level.upper() == 'DEBUG'))
            callback_manager = CallbackManager([debug_handler])

            self.pubmed_tool = PubMedTool(
                email=env_values["ENTREZ_EMAIL"],
                api_key=env_values["ENTREZ_API_KEY"]
            )
            self.analyzer = LiteratureAnalyzer()
            self.vector_db_url = vector_db_url

            # 使用全局LLM
            self.llm = Settings.llm
            if self.llm is None:
                raise RuntimeError("未配置全局LLM实例")

            # 创建聊天记忆
            self.memory = ChatMemoryBuffer.from_defaults(token_limit=4096)

            embeddings = Settings.embed_model
            if not isinstance(embeddings, OpenAIEmbedding):
                raise RuntimeError("未配置正确的OpenAI嵌入模型实例")

            # 初始化向量存储
            self.literature_store = PubMedVectorStore(
                connection_string=self.vector_db_url,
                embed_model=embeddings
            )
            self.analysis_store = AnalysisVectorStore(
                connection_string=self.vector_db_url,
                embed_model=embeddings
            )

            # 创建工具列表
            tools = [
                FunctionTool.from_defaults(
                    fn=self.search_literature,
                    name="search_literature",
                    description="""在线搜索PubMed文献资料。
                    输入参数：
                    - compound: 化合物名称
                    - disease: 疾病名称
                    - start_year: 起始年份（可选）
                    - end_year: 结束年份（可选）
                    """
                ),
                FunctionTool.from_defaults(
                    fn=self.analyze_literature,
                    name="analyze_literature",
                    description="""分析文献内容并生成报告。
                    输入参数：
                    - compound: 化合物名称
                    - disease: 疾病名称
                    - pmids: PubMed文献ID列表
                    """
                ),
                FunctionTool.from_defaults(
                    fn=self.query_previous_analyses,
                    name="query_previous_analyses",
                    description="""查询历史分析结果。
                    输入参数：
                    - query: 查询文本
                    - compound: 过滤特定化合物（可选）
                    - disease: 过滤特定疾病（可选）
                    - limit: 返回结果数量（可选）
                    """
                ),
                FunctionTool.from_defaults(
                    fn=self.search_similar_articles,
                    name="search_similar_articles",
                    description="""搜索相似文献，可以提供关键词查询和过滤条件。
                    输入参数：
                    - query: 搜索查询
                    - compound: 过滤特定化合物（可选）
                    - disease: 过滤特定疾病（可选）
                    - limit: 返回结果数量（可选）
                    """
                )
            ]

            # 初始化ReAct Agent
            self.agent = ReActAgent.from_tools(
                tools=cast(List[BaseTool], tools),
                llm=self.llm,
                memory=self.memory,
                callback_manager=callback_manager,
                verbose=True,
                system_prompt="""你是一个专业的医学文献分析助手。你的主要任务是帮助用户分析化合物对特定疾病的治疗效果。

工作流程：
1. 理解用户的查询意图
2. 首先查询历史分析结果，避免重复分析
3. 如果没有相关的历史结果，再执行新的分析：
   - 搜索最新的相关文献
   - 分析文献内容
   - 生成分析报告：对于每个化合物，提供治疗效果整体结论；并并列出相关文献的PMID、关键发现的归纳。
4. 整合所有信息并提供建议
5. 与用户进行多轮对话，澄清需求或提供补充信息

注意事项：
- 优先使用历史分析结果，避免重复工作
- 主动提出建议和补充问题
- 解释每个分析步骤
- 使用专业且易懂的语言
- 保持对话的连贯性

工具使用说明：
1. query_previous_analyses: 查询历史分析结果，应该首先使用此工具
2. search_literature: 用于搜索PubMed文献，需要提供化合物名称和疾病名称
3. analyze_literature: 分析文献内容，需要提供化合物名称、疾病名称和文献ID列表
4. search_similar_articles: 搜索相似文献，可以提供关键词查询和过滤条件

示例查询：
用户: "帮我分析槲皮素对肺癌的疗效"
思考: 我应该先查看是否有历史分析结果，如果没有再进行新的分析。
行动1: 使用query_previous_analyses工具查询历史结果
行动2: 如果没有相关结果，使用search_literature工具搜索新文献
行动3: 使用analyze_literature工具分析文献内容
""",
                task_prompt="请帮助用户分析化合物对疾病的治疗效果。优先查询历史分析结果，如果没有相关结果再进行新的文献分析。"
            )

            # 设置调试处理器的详细程度
            self.debug_handler = debug_handler

        except Exception as e:
            logger.error("初始化代理失败: %s", e, exc_info=True)
            raise RuntimeError(f"初始化代理失败: {e}") from e

    def search_literature(self, compound: str, disease: str,
                          start_year: Optional[int] = None,
                          end_year: Optional[int] = None) -> List[str]:
        """搜索相关文献

        Args:
            compound (str): 化合物名称
            disease (str): 疾病名称
            start_year (Optional[int]): 开始年份
            end_year (Optional[int]): 结束年份

        Returns:
            List[str]: 文献ID列表
        """
        logger.info("搜索文献: compound=%s, disease=%s, start_year=%s, end_year=%s",
                    compound, disease, start_year, end_year)
        try:
            return self.pubmed_tool.search_pubmed(
                compound, disease, start_year=start_year, end_year=end_year
            )
        except Exception as e:
            logger.error("搜索文献失败: %s", e, exc_info=True)
            raise

    def analyze_literature(self, compound: str, disease: str, pmids: List[str]) -> Dict[str, Any]:
        """分析文献内容

        Args:
            compound (str): 化合物名称
            disease (str): 疾病名称
            pmids (List[str]): 文献ID列表

        Returns:
            Dict[str, Any]: 分析报告（JSON格式）
        """
        logger.info("分析文献: compound=%s, disease=%s, pmids=%s",
                    compound, disease, pmids)
        try:
            # 获取文献详情
            articles = self.pubmed_tool.fetch_articles(pmids)
            for article in articles:
                # 存储文献
                try:
                    self.literature_store.store_article(
                        article=article,
                        compound=compound,
                        disease=disease
                    )
                except Exception as e:
                    logger.error("存储文献失败: %s", str(e), exc_info=True)
                    # 继续执行，不影响分析

            # 分析每篇文献
            analyses = []
            for article in articles:
                analysis = self.analyzer.analyze_article(
                    compound, disease, article)
                analyses.append(analysis)

            # 统计分析结果
            positive = sum(1 for a in analyses if a.get(
                'effect_type') == 'positive')
            negative = sum(1 for a in analyses if a.get(
                'effect_type') == 'negative')
            neutral = sum(1 for a in analyses if a.get(
                'effect_type') == 'neutral')

            # 提取关键发现
            key_findings = []
            for analysis in analyses:
                if 'findings' in analysis:
                    key_findings.extend(analysis['findings'])

            # 生成报告
            report = {
                'total_papers': len(articles),
                'positive_effects': positive,
                'negative_effects': negative,
                'neutral_effects': neutral,
                'key_findings': key_findings[:5],  # 取前5个关键发现
                'articles': [{
                    'pmid': article.pmid,
                    'title': article.title,
                    'conclusion': article.conclusion,
                    'effect_type': analysis.get('effect_type'),
                    'evidence_strength': analysis.get('evidence_strength')
                } for article, analysis in zip(articles, analyses)]
            }

            # 存储分析结果
            try:
                content = f"{compound}对{disease}的分析报告：\n" + \
                    f"总计分析文献：{len(articles)}篇\n" + \
                    f"正面效果：{positive}篇，负面效果：{negative}篇，中性效果：{neutral}篇\n" + \
                    f"主要发现：{', '.join(key_findings[:5])}"

                self.analysis_store.store_analysis(
                    compound=compound,
                    disease=disease,
                    content=content
                )
                logger.info("成功存储分析结果")
            except Exception as e:
                logger.error("存储分析结果失败: %s", str(e), exc_info=True)
                # 继续执行，不影响主流程

            return report
        except Exception as e:
            logger.error("分析文献失败: %s", e, exc_info=True)
            raise

    def query_previous_analyses(self, compound: str, disease: str) -> List[Dict[str, Any]]:
        """查询历史分析结果

        Args:
            compound (Optional[str]): 过滤特定化合物
            disease (Optional[str]): 过滤特定疾病

        Returns:
            List[Dict[str, Any]]: 相关的历史分析结果，Json格式
        """
        try:
            results = self.analysis_store.get_analysis_history(
                compound=compound,
                disease=disease
            )

            # 去掉向量字段
            results = [{k: v for k, v in analysis.__dict__.items()
                        if k != 'content_vector'} for analysis in results]

            return [analysis for analysis in results]

        except Exception as e:
            logger.error("查询历史分析失败: %s", str(e), exc_info=True)
            return []

    def search_similar_articles(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """搜索相似文献

        Args:
            query (str): 搜索查询
            limit (int): 返回结果数量

        Returns:
            List[Dict[str, Any]]: 相似文献列表
        """
        try:
            results = self.literature_store.hybrid_search(
                query=query,
                limit=limit
            )

            return [{
                'pmid': article.pmid,
                'title': article.title,
                'abstract': article.abstract,
                'compound': article.compound,
                'disease': article.disease,
                'similarity': float(score)
            } for article, score in results]

        except Exception as e:
            logger.error("搜索相似文献失败: %s", str(e), exc_info=True)
            return []

    def chat(self, message: str) -> str:
        """处理用户消息，支持多轮对话

        Args:
            message (str): 用户消息

        Returns:
            str: 代理响应或错误信息
        """
        logger.info("处理用户消息: message=%s", message)
        if not message or not message.strip():
            return "消息不能为空"

        try:
            # 打印开始处理的提示
            print("\n=== 开始处理查询 ===")
            print(f"用户输入: {message}")
            print("\n=== 执行步骤 ===")

            # 执行查询
            response = self.agent.chat(message)

            # 打印执行追踪
            print("\n=== 执行追踪 ===")
            self.debug_handler.print_trace_on_end = True

            return response
        except Exception as e:
            logger.error("处理消息时出错: %s", e, exc_info=True)
            error_msg = f"处理消息时出错: {e}"
            print(f"\n=== 错误信息 ===\n{error_msg}")
            return error_msg

    def reset(self):
        """重置对话历史"""
        logger.info("重置对话历史")
        self.memory.reset()
