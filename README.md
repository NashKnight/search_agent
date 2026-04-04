# Search Agent

基于 vLLM + Jina Search 构建的多轮搜索智能体。通过结构化的**记忆板（Memory Board）**跨轮次追踪任务目标、已收集信息和待搜索队列，实现自主的迭代式信息收集与问答。

---

## 项目结构

```
search_agent/
├── config.yaml               # 全局配置（模型路径、API Key、限制参数、裁判配置等）
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
├── infer.py                  # 推理脚本：运行模型，将预测结果写入 tests/
├── eval.py                   # 评分脚本：调用裁判模型，计算 accuracy
└── tests/
    ├── smoke_test.jsonl      # 冒烟测试用例（6 条，无标准答案，用于快速验证流程）
    └── run_<时间戳>.jsonl    # infer.py 输出的预测结果
```

---

## 各组件说明

| 组件 | 职责 |
|---|---|
| `config.yaml` | 统一配置入口：模型路径、Jina API Key、代理、Token 限制、裁判 API |
| `utils/config_loader.py` | `load_config(path?)` — 加载 YAML，返回 dict |
| `models/base.py` | `BaseLLM` 抽象类 — 定义 `generate()` 和 `clear_cache()` 接口 |
| `models/vllm_model.py` | `VLLMModel` — 通过 vLLM 加载本地模型，应用 chat template |
| `search/base.py` | `BaseSearch` 抽象类 — 定义 `search(query, max_results)` 接口 |
| `search/jina_search.py` | `JinaSearch` — 调用 Jina Search API，解析 JSON，支持代理 |
| `agent/memory.py` | `MemoryManager` — `initialize()` / `update()` / `get()` / `reset()` |
| `agent/prompts.py` | 所有 Prompt 以命名常量形式集中存放 |
| `search_workflow.py` | `SearchWorkflow.run(query)` — 编排完整多轮搜索流程 |
| `infer.py` | 推理入口：遍历 benchmark，保存原始预测结果到 JSONL |
| `eval.py` | 评分入口：读取预测 JSONL，逐条调用裁判模型，输出 accuracy |
| `tests/smoke_test.jsonl` | 6 条冒烟测试用例，用于快速验证模型能否正常运行 |

---

## 环境准备

### 1. 安装依赖

```bash
pip install vllm transformers requests pyyaml
```

### 2. 修改 `config.yaml`

**推理所需（必填）：**

```yaml
model:
  local_model_path: "/path/to/your/model"

search:
  jina_api_key: "jina_xxxxxxxxxxxxxxxxxxxx"
  use_proxy: false
```

**评分所需（必填）：**

```yaml
judge:
  api_url: "https://api.openai.com/v1"   # 任意 OpenAI 兼容接口
  api_key: "sk-xxxxxxxxxxxxxxxxxxxx"
  model: "gpt-4o"                         # 或 qwen-max 等
```

---

## 使用方法

所有命令均在 **`search_agent/` 目录内**运行。

---

### Step 1：推理（infer.py）

运行模型，将预测结果写入 `tests/` 目录。

```bash
# 使用冒烟测试集快速验证（6 条，无标准答案）
python infer.py --benchmark tests/smoke_test.jsonl

# 跑完整 WebWalker benchmark（路径从 config.yaml 读取）
python infer.py

# 只跑前 N 条
python infer.py --limit 10

# 跳过前 N 条（断点续跑）
python infer.py --offset 50 --limit 20

# 指定 benchmark 文件
python infer.py --benchmark ../benchmark/webwalker/main-00000-of-00001.jsonl

# 自定义输出路径
python infer.py --output tests/my_run.jsonl

# 组合参数
python infer.py \
  --benchmark ../benchmark/webwalker/main-00000-of-00001.jsonl \
  --limit 50 \
  --offset 0 \
  --output tests/webwalker_50.jsonl
```

---

### Step 2：评分（eval.py）

读取 infer.py 生成的 JSONL，调用裁判模型逐条打分，输出 accuracy。

```bash
# 评分指定的预测文件（结果保存为同名 _eval.json）
python eval.py --input tests/run_20240101_120000.jsonl

# 自定义输出路径
python eval.py --input tests/run_20240101_120000.jsonl --output tests/result.json

# 调整并发数（默认 4，加快评分速度）
python eval.py --input tests/run_20240101_120000.jsonl --concurrency 8

# 使用不同的 config（换裁判模型）
python eval.py --input tests/run.jsonl --config config.yaml
```

---

### 单条查询（Python 调用）

```python
from models.vllm_model import VLLMModel
from search.jina_search import JinaSearch
from search_workflow import SearchWorkflow

llm      = VLLMModel()
searcher = JinaSearch()
workflow = SearchWorkflow(llm=llm, searcher=searcher)

result = workflow.run("EU4Health 资助的 SUPPLY 项目是什么时候开始的？")

print(result["answer"])        # 最终答案
print(result["memory"])        # 最终记忆板状态
print(result["used_sources"])  # {url: title} 引用来源
print(result["rounds"])        # 每轮详细 trace
```

---

## 文件格式说明

### infer.py 输出（`tests/run_*.jsonl`）

每行一条预测结果：

```json
{
  "question": "查询特斯拉的实时股价",
  "gold_answer": "...",
  "predicted_answer": "截至今日，特斯拉（TSLA）股价为 ...",
  "used_sources": {"https://...": "Tesla Stock Price"},
  "final_memory": "[User Goal]\n...\n[Collected Information]\n...",
  "num_rounds": 3,
  "root_url": "",
  "info": {"domain": "finance", "difficulty_level": "easy"},
  "error": null
}
```

### eval.py 输出（`tests/*_eval.json`）

```json
{
  "summary": {
    "total": 100,
    "correct": 72,
    "incorrect": 24,
    "unknown": 4,
    "evaluated": 96,
    "accuracy": 0.75,
    "accuracy_pct": "75.00%",
    "judge_model": "gpt-4o"
  },
  "results": [ ... ]
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

### 更换裁判模型

只需修改 `config.yaml` 的 `judge` 块，指向任意 OpenAI-compatible 接口即可，代码无需改动。
