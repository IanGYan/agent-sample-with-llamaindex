# 医学文献智能分析系统

基于 LlamaIndex 和 OpenAI 的智能文献分析系统，用于分析化合物对疾病的治疗效果。
本项目仅仅为了学习演示，不能用于生产环境。

## 功能特点

- 🔍 支持PubMed文献智能检索
- 🤖 基于LLM的文献内容深度分析
- 📊 自动生成分析报告
- 💾 向量化存储分析历史
- 🗣️ 自然语言交互界面

## 系统架构

### 核心组件

1. **文献检索工具** (`tools/pubmed_tool.py`)
   - 基于PubMed API的文献检索
   - 支持年份范围过滤
   - 自动获取文献详细信息

2. **文献分析工具** (`tools/literature_analyze.py`)
   - 使用OpenAI进行内容分析
   - 提取关键发现和结论
   - 评估证据强度

3. **智能体** (`agent/llama_agent.py`)
   - 基于LlamaIndex的智能交互
   - 自动任务分解和执行
   - 结果整合和报告生成

4. **数据模型** (`models/pubmed_models.py`)
   - 规范的数据结构定义
   - 基于Pydantic的数据验证

5. **向量存储** (`memory/vector_store.py`)
   - 使用pgvector存储分析结果
   - 支持相似性检索
   - 历史记录管理

## 安装说明

### 环境要求

- Python 3.8+
- PostgreSQL（带pgvector扩展）
- OpenAI API密钥
- PubMed API访问凭证

### 安装步骤

1. 克隆仓库：

   ```bash
   git clone [repository_url]
   cd medical-literature-analysis
   ```

2. 安装依赖：

   ```bash
   pip install -r requirements.txt
   ```

3. 配置环境变量:

   - 复制 `.env.sample` 为 `.env`
   - 填写以下配置：

     ```text
     ENTREZ_EMAIL=your.email@example.com
     ENTREZ_API_KEY=your_pubmed_api_key
     OPENAI_API_KEY=your_openai_api_key
     POSTGRES_CONNECTION=postgresql://user:password@localhost:5432/dbname
     ```

     注意：数据库`dbname`请自行创建

## 使用说明

### 交互式模式

```bash
python main.py --interactive
```

在交互式模式下，您可以直接输入自然语言查询，例如：

- "帮我分析Quercetin对Lung Cancer的疗效"
- "分析2020年以来Vitamin D对Depression的研究证据"

### 命令行模式

```bash
python main.py --query "帮我分析Quercetin对Lung Cancer的疗效"
```

## 分析报告示例

系统将生成包含以下内容的分析报告：

- 文献统计信息
- 效果评估结果
- 关键研究发现
- 证据强度分析
- 详细参考文献

## 贡献指南

欢迎提交Issue和Pull Request！

## 许可证

MIT License
