# Search Agent

基于 vLLM + Jina Search 构建的多轮搜索智能体。通过结构化的**记忆板（Memory Board）**跨轮次追踪任务目标、已收集信息和待搜索队列，实现自主的迭代式信息收集与问答。

---

## 项目结构

```
search_agent/
├── config.yaml               # 全局配置（模型路径、API Key、限制参数、评测路径）
├── __init__.py
│
├── utils/
│   ├── config_loader.py      # load_config() — 读取 config.yaml 返回 dict
│   └── __init__.py
│
├── models/
│   ├── base.py               # 抽象基类 BaseLLM
│   ├── vllm_model.py         # vLLM 实现（本地模型推理）
│   └── __init__.py
│
├── search/
│   ├── base.py               # 抽象基类 BaseSearch
│   ├── jina_search.py        # Jina Search API 实现
│   └── __init__.py
│
├── agent/
│   ├── memory.py             # MemoryManager — 跨轮次维护全局记忆板
│   ├── prompts.py            # 所有 Prompt 模板集中管理
│   └── __init__.py
│
├── search_workflow.py        # SearchWorkflow — 多轮主循环逻辑
├── eval.py                   # Benchmark 评测入口（CLI 脚本）
└── tests/                    # 评测结果输出目录（自动生成 run_<时间戳>.jsonl）
```

---

## 各组件说明

| 组件 | 职责 |
|---|---|
| `config.yaml` | 统一配置入口：模型路径、Jina API Key、代理、Token 限制、评测路径 |
| `utils/config_loader.py` | `load_config(path?)` — 加载 YAML，返回 dict，通过依赖注入传递给各模块 |
| `models/base.py` | `BaseLLM` 抽象类 — 定义 `generate(prompt, max_new_tokens)` 和 `clear_cache()` 接口 |
| `models/vllm_model.py` | `VLLMModel` — 通过 vLLM 加载本地模型，应用 chat template，返回 `(token_ids, raw, clean)` |
| `search/base.py` | `BaseSearch` 抽象类 — 定义 `search(query, max_results)` 接口，返回 `{sources, error}` |
| `search/jina_search.py` | `JinaSearch` — 调用 Jina Search API，解析 JSON，支持代理配置 |
| `agent/memory.py` | `MemoryManager` — 提供 `initialize()` / `update()` / `get()` / `reset()` 管理记忆板状态 |
| `agent/prompts.py` | 所有 Prompt 以命名常量形式集中存放（`BASE_PROMPT`、`MEMORY_INIT_PROMPT`、`FILTER_QUERIES_PROMPT` 等） |
| `search_workflow.py` | `SearchWorkflow.run(query)` — 编排 Round 1 → 搜索循环 → 强制生成最终答案的完整流程 |
| `eval.py` | CLI 脚本：加载 JSONL benchmark，逐题运行，实时将结果流式写入 `tests/` |

---

## 环境准备

### 1. 安装依赖

```bash
pip install vllm transformers requests pyyaml
```

### 2. 修改 `config.yaml`

至少需要配置模型路径和 Jina API Key：

```yaml
model:
  local_model_path: "/path/to/your/model"   # 例如 /root/autodl-tmp/Qwen3-8B

search:
  jina_api_key: "jina_xxxxxxxxxxxxxxxxxxxx"
  use_proxy: false   # 如需代理，改为 true 并填写 proxies 字段
```

---

## 使用方法

所有命令均在 `search_agent/` 的**父目录**（即 `deep_research/`）下运行。

### 单条查询（Python 调用）

```python
from search_agent.models.vllm_model import VLLMModel
from search_agent.search.jina_search import JinaSearch
from search_agent.search_workflow import SearchWorkflow

llm      = VLLMModel()
searcher = JinaSearch()
workflow = SearchWorkflow(llm=llm, searcher=searcher)

result = workflow.run("EU4Health 资助的 SUPPLY 项目是什么时候开始的？")

print(result["answer"])          # 最终答案
print(result["memory"])          # 最终记忆板状态
print(result["used_sources"])    # {url: title} 引用来源
print(result["rounds"])          # 每轮详细 trace
```

### 运行 Benchmark 评测（默认配置）

评测 `config.yaml → eval.benchmark_path` 中指定的全量 benchmark，
结果保存至 `search_agent/tests/run_<时间戳>.jsonl`。

```bash
python -m search_agent.eval
```

### 只评测前 N 条

```bash
python -m search_agent.eval --limit 10
```

### 跳过前 N 条（断点续评）

```bash
python -m search_agent.eval --offset 50 --limit 20
```

### 指定不同的 benchmark 文件

```bash
python -m search_agent.eval --benchmark /path/to/other_benchmark.jsonl
```

### 自定义输出文件路径

```bash
python -m search_agent.eval --output search_agent/tests/my_run.jsonl
```

### 指定不同的配置文件

```bash
python -m search_agent.eval --config /path/to/other_config.yaml
```

### 组合参数示例

```bash
python -m search_agent.eval \
  --config search_agent/config.yaml \
  --benchmark ../benchmark/webwalker/main-00000-of-00001.jsonl \
  --limit 50 \
  --offset 0 \
  --output search_agent/tests/webwalker_50.jsonl
```

---

## 输出格式

`tests/` 目录下每个 JSONL 文件，每行为一条评测结果：

```json
{
  "question": "EU4Health 资助的 SUPPLY 项目是什么时候开始的？",
  "gold_answer": "EU4Health 项目，2022年9月1日开始，持续18个月。",
  "predicted_answer": "SUPPLY 项目由 EU4Health 计划资助，于 2022 年 9 月 1 日启动……",
  "used_sources": {"https://ehaweb.org/...": "EHA News"},
  "final_memory": "[User Goal]\n...\n[Collected Information]\n...",
  "num_rounds": 4,
  "root_url": "https://ehaweb.org/",
  "info": {"domain": "conference", "difficulty_level": "medium"},
  "error": null
}
```

---

## 扩展指南

### 新增 LLM 后端

1. 在 `models/` 下新建文件，继承 `BaseLLM`
2. 实现 `generate()` 和 `clear_cache()`
3. 实例化后传入 `SearchWorkflow(llm=YourModel(), ...)`

### 新增搜索后端

1. 在 `search/` 下新建文件，继承 `BaseSearch`
2. 实现 `search()`，返回 `{"sources": {...}, "error": None}`
3. 实例化后传入 `SearchWorkflow(searcher=YourSearch(), ...)`

### 修改 Prompt

所有 Prompt 模板集中在 `agent/prompts.py`，直接修改对应常量即可，无需改动其他文件。
