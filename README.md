# Search Agent

基于 vLLM + Jina Search 构建的多轮搜索智能体。通过结构化的**记忆板（Memory Board）**跨轮次追踪任务目标、已收集信息和待搜索队列，实现自主的迭代式信息收集与问答。

---

## 项目结构

```
search_agent/
├── config.yaml               # 全局配置（模型路径、vLLM server 端口、API Key、限制参数、裁判配置等）
├── start_vllm.sh             # 启动推理 vLLM server（端口 6001，单卡）
├── start_judge_vllm.sh       # 启动裁判 vLLM server（端口 6002，双卡 TP=2）
├── __init__.py
│
├── utils/
│   ├── config_loader.py      # load_config() — 读取 config.yaml 返回 dict
│   └── __init__.py
│
├── models/
│   ├── base.py               # 抽象基类 BaseLLM
│   ├── vllm_model.py         # vLLM 进程内模式（本地直接加载，备用）
│   ├── vllm_server_model.py  # vLLM server 模式（HTTP API，多线程推荐）
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
├── infer.py                  # 推理脚本：多线程并行推理，结果按原始顺序写入 JSONL
├── eval.py                   # 评分脚本：调用裁判模型，计算 accuracy
└── tests/
    ├── smoke_test.jsonl      # 冒烟测试用例（6 条，无标准答案，用于快速验证流程）
    └── run_<时间戳>.jsonl    # infer.py 输出的预测结果
```

---

## 各组件说明

| 组件 | 职责 |
|---|---|
| `config.yaml` | 统一配置入口：模型路径、vLLM server 端口、Jina API Key、代理、Token 限制、裁判 API |
| `start_vllm.sh` | 启动推理 vLLM server（端口 6001），支持 `--port`、`--gpu`、`--tp` 等参数 |
| `start_judge_vllm.sh` | 启动裁判 vLLM server（端口 6002，默认双卡 TP=2），读取 `judge.model_path` |
| `utils/config_loader.py` | `load_config(path?)` — 加载 YAML，返回 dict |
| `models/base.py` | `BaseLLM` 抽象类 — 定义 `generate()` 和 `clear_cache()` 接口 |
| `models/vllm_model.py` | `VLLMModel` — 进程内直接加载模型（备用，单线程场景） |
| `models/vllm_server_model.py` | `VLLMServerModel` — 通过 HTTP API 连接 vLLM server（多线程推荐） |
| `search/base.py` | `BaseSearch` 抽象类 — 定义 `search(query, max_results)` 接口 |
| `search/jina_search.py` | `JinaSearch` — 调用 Jina Search API，解析 JSON，支持代理 |
| `agent/memory.py` | `MemoryManager` — `initialize()` / `update()` / `get()` / `reset()` |
| `agent/prompts.py` | 所有 Prompt 以命名常量形式集中存放 |
| `search_workflow.py` | `SearchWorkflow.run(query)` — 编排完整多轮搜索流程 |
| `infer.py` | 推理入口：多线程并行推理，tqdm 进度条，结果按原始顺序写入 JSONL |
| `eval.py` | 评分入口：读取预测 JSONL，逐条调用裁判模型，输出 accuracy |
| `tests/smoke_test.jsonl` | 6 条冒烟测试用例，用于快速验证模型能否正常运行 |

---

## 环境准备

### 1. 安装依赖

```bash
pip install vllm transformers requests pyyaml openai
```

### 2. 修改 `config.yaml`

**推理所需（必填）：**

```yaml
model:
  local_model_path: "/path/to/your/model"

vllm_server:
  host: "127.0.0.1"
  port: 6001          # 与 start_vllm.sh --port 保持一致

search:
  jina_api_key: "jina_xxxxxxxxxxxxxxxxxxxx"
  use_proxy: false
```

**评分所需（必填）：**

```yaml
judge:
  model_path: "/path/to/judge/model"     # 裁判模型路径，供 start_judge_vllm.sh 使用
  api_url: "http://127.0.0.1:6002/v1"   # 本地裁判 vLLM（start_judge_vllm.sh 启动）
  api_key: "EMPTY"                       # 本地 vLLM 不需要真实 key
  model: "your-judge-model-name"         # /v1/models 返回的模型名
```

---

## 推理工作流详解

调用 `python infer.py` 后，内部执行链路如下：

```
infer.py
│
├─ 1. 读取 config.yaml，加载 benchmark JSONL
│
├─ 2. 初始化组件
│     ├─ VLLMServerModel ← 连接 vLLM server（HTTP API，需先运行 start_vllm.sh）
│     ├─ JinaSearch      ← 初始化 Jina Search 客户端
│     └─ SearchWorkflow  ← 注入上面两个组件
│
└─ 3. 多线程（ThreadPoolExecutor）并行处理每条 question
       ├─ 每个线程独立调用 SearchWorkflow.run(query)
       ├─ 结果收集到 dict[original_idx -> result]（保序）
       └─ 全部完成后按原始顺序写入 JSONL
       │
       ├─ [Round 1] 初始分析
       │     ├─ 将 question 拼入 BASE_PROMPT，发给模型
       │     ├─ 解析模型输出中的 <search>...</search> 标签，提取待搜索词
       │     ├─ 若无 <search>，说明无需搜索 → 直接返回模型答案，结束
       │     ├─ 调用 MemoryManager.initialize() 创建记忆板（含用户目标、任务规划、待搜索队列）
       │     └─ 调用过滤器（模型二次判断）筛掉与目标无关的查询，剩余入队
       │
       ├─ [Round 2~N] 搜索循环（直到队列为空或达到 max_rounds）
       │     ├─ 从队列取出一条查询词
       │     ├─ 调用 JinaSearch.search() → 请求 Jina Search API，返回最多 5 条结果
       │     ├─ 过滤器再次判断本条查询是否仍与目标相关
       │     │     └─ 不相关 → 跳过，不调用模型，不更新记忆板
       │     ├─ 将搜索结果格式化，拼入 ANALYSIS_PROMPT，发给模型分析
       │     ├─ 解析模型输出：
       │     │     ├─ 有新 <search> → 追加到待处理列表
       │     │     └─ 无 <search>  → 模型认为信息已足够，本轮即为最终答案
       │     ├─ 合并（旧队列 + 新查询），整体过滤一次，重建队列
       │     └─ 调用 MemoryManager.update() 将本轮搜索结果和新队列写入记忆板
       │
       └─ [Final] 强制生成最终答案（队列耗尽或达到 max_rounds 时触发）
             ├─ 将完整记忆板拼入 FINAL_ANSWER_PROMPT，发给模型汇总
             └─ 清理 <think>/<search> 标签，返回干净的最终答案

infer.py 收到 SearchWorkflow.run() 的返回值后：
  └─ 将 {question, gold_answer, predicted_answer, used_sources, final_memory, num_rounds} 写入 JSONL
```

**关键数据流：**

```
question
  └→ [模型] 初始分析 → <search> 列表
       └→ [记忆板] 初始化
            └→ [过滤器] 筛选查询 → 队列
                 └→ 循环：[Jina API] 搜索 → [模型] 分析 → 更新队列 + 记忆板
                      └→ [模型] 最终汇总 → answer
```

---

## 使用方法

所有命令均在 **`search_agent/` 目录内**运行。

---

### Step 1：启动 vLLM server（start_vllm.sh）

推理前需先单独启动 vLLM server，脚本会等待端口就绪后才退出等待循环。

```bash
# 基础用法：单卡，端口 6001（从 config.yaml 读取模型路径）
bash start_vllm.sh --gpu 0

# 指定端口和模型路径
bash start_vllm.sh --port 6001 --gpu 0 --model /root/autodl-tmp/Qwen3-8B

# 多卡 Tensor Parallel
bash start_vllm.sh --port 6001 --gpu 0,1 --tp 2

# 后台运行，日志写文件
nohup bash start_vllm.sh --port 6001 --gpu 0 > vllm_6001.log 2>&1 &
```

**参数说明：**

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--port` | 监听端口 | `config.yaml` 中 `vllm_server.port`，否则 `6001` |
| `--model` | 模型路径 | `config.yaml` 中 `model.local_model_path` |
| `--gpu` | `CUDA_VISIBLE_DEVICES` | `0` |
| `--tp` | `tensor-parallel-size` | `1` |
| `--timeout` | 等待就绪超时（秒） | `600` |
| `--extra` | 透传给 `vllm serve` 的额外参数 | — |

---

### Step 2：推理（infer.py）

连接已启动的 vLLM server，多线程并行推理，结果按原始输入顺序写入 JSONL。

```bash
# 使用冒烟测试集快速验证（6 条，无标准答案）
python infer.py --benchmark tests/smoke_test.jsonl

# 跑完整 WebWalker benchmark（路径从 config.yaml 读取）
python infer.py

# 开 8 个并发线程
python infer.py --workers 8

# 指定 vLLM server 端口（覆盖 config.yaml）
python infer.py --port 6002

# 只跑前 N 条
python infer.py --limit 10

# 跳过前 N 条（断点续跑）
python infer.py --offset 50 --limit 20

# 组合参数
python infer.py \
  --benchmark ../benchmark/webwalker/main-00000-of-00001.jsonl \
  --port 6001 \
  --workers 8 \
  --output tests/webwalker_run.jsonl
```

---

### Step 3：评分（eval.py）

先启动裁判 vLLM server，再运行 eval.py。裁判会结合问题、标准答案、预测答案和参考 URL 综合评判对错。

```bash
# 1. 启动裁判 vLLM（双卡，端口 6002，后台运行）
bash start_judge_vllm.sh -d

# 2. 评分指定的预测文件（结果保存为同名 _eval.json）
python eval.py --input tests/run_20240101_120000.jsonl

# 自定义输出路径
python eval.py --input tests/run_20240101_120000.jsonl --output tests/result.json

# 调整并发数（默认 8）
python eval.py --input tests/run_20240101_120000.jsonl --concurrency 16

# 使用不同的 config
python eval.py --input tests/run.jsonl --config config.yaml
```

---

### 单条查询（Python 调用）

```python
# 需先启动 vLLM server：bash start_vllm.sh --port 6001 --gpu 0
from models.vllm_server_model import VLLMServerModel
from search.jina_search import JinaSearch
from search_workflow import SearchWorkflow

llm      = VLLMServerModel(port=6001)   # 连接 vLLM server
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

修改 `config.yaml` 的 `judge` 块：`model_path` 指向新模型路径，`model` 填对应名称，重启 `start_judge_vllm.sh` 即可，代码无需改动。
