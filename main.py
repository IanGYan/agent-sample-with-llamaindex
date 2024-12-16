"""主程序入口

本模块提供了文献分析代理的主要入口点。
它负责初始化所有必要的组件并启动代理服务。

依赖:
    - openai: OpenAI API接口
    - llama_index_core: 用于文档索引和查询
    - pgvector: PostgreSQL向量存储扩展
"""

import os
import logging
import argparse
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from agent.pubmed_agent import MedLiteratureAnalysisAgent

# 配置日志
log_level = os.getenv('LOG_LEVEL', 'ERROR')  # 如果未设置，默认使用 ERROR
numeric_level = getattr(logging, log_level.upper(), logging.ERROR)

logging.basicConfig(
    level=numeric_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """主程序入口点
    """
    try:
        # 检查必要的环境变量
        required_env_vars = {
            "OPENAI_API_KEY": "OpenAI API密钥",
            "ENTREZ_EMAIL": "PubMed API邮箱",
            "ENTREZ_API_KEY": "PubMed API密钥",
            "PGVECTOR_URL": "向量数据库连接字符串",
            "DEFAULT_MODEL_NAME": "默认模型名称",
            "OPENAI_EMBEDDING_MODEL_NAME": "OpenAI嵌入模型名称"
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

        # 加载环境变量
        load_dotenv()

        # 解析命令行参数
        parser = argparse.ArgumentParser(description='医学文献分析系统')
        parser.add_argument('--interactive', action='store_true', help='启动交互式会话')
        parser.add_argument('--query', type=str, help='直接处理单个查询')
        args = parser.parse_args()

        # 初始化 OpenAI
        llm = OpenAI(
            model=env_values["DEFAULT_MODEL_NAME"],
            api_key=env_values["OPENAI_API_KEY"],
        )
        embed_model = OpenAIEmbedding(
            model=env_values["OPENAI_EMBEDDING_MODEL_NAME"],
            deployment=1536,
            api_key=env_values["OPENAI_API_KEY"]
        )

        # 设置全局LLM和嵌入模型
        Settings.llm = llm
        Settings.embed_model = embed_model

        # 初始化 Agent
        agent = MedLiteratureAnalysisAgent(
            vector_db_url=env_values["PGVECTOR_URL"]
        )
        logger.info("成功初始化文献分析代理")

        if args.interactive:
            print("欢迎使用医学文献分析系统！")
            print("输入 'quit' 或 'exit' 退出")

            while True:
                query = input("\n请输入您的问题: ")
                if query.lower() in ['quit', 'exit']:
                    break

                try:
                    response = agent.chat(query)
                    print("\n分析结果:")
                    print(response)
                except Exception as e:
                    print(f"\n处理查询时出错: {str(e)}")

        elif args.query:
            try:
                response = agent.chat(args.query)
                print(response)
            except Exception as e:
                print(f"处理查询时出错: {str(e)}")
        else:
            parser.print_help()

    except Exception as e:
        logger.error("程序启动失败: %s", e, exc_info=True)
        raise

if __name__ == "__main__":
    main()
