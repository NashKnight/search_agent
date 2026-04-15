# Search Agent

基于 vLLM + Jina Search 构建的多轮搜索智能体，支持结构化 Dynamic Memory 与多跳推理。

---

## 项目结构

```
search_agent/
├── config.yaml               # 全局配置（模型路径、vLLM server 端口、API Key、限制参数、裁判配置等）
│
├── commands/
│   ├── start_vllm.sh         # 启动推理 vLLM server（端口 6001，单卡）
│   └── start_judge_vllm.sh   # 启动裁判 vLLM server（端口 6002，双卡 TP=2）
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
│   ├── memory.py             # MemoryManager — 跨轮次维护 Dynamic Memory
│   ├── prompts.py            # 所有 prompt 模板（中文）
│   └── __init__.py
│
├── search_workflow.py        # SearchWorkflow — 核心推理循环（多轮搜索 + Dynamic Memory）
├── infer.py                  # 主推理脚本：调用 SearchWorkflow，并发处理 benchmark
├── eval.py                   # 评分脚本：调用裁判模型，计算 accuracy
│
├── baselines/
│   ├── infer_react.py        # 基线推理：Vanilla ReAct（Thought/Action/Observation 循环）
│   ├── infer_base.py         # 基线推理：纯 LLM 直接问答 / 单跳 Jina 搜索（--jina）
│   └── infer_webdancer.py    # 基线推理：WebDancer 兼容（Serper 搜索 + Jina Reader）
│
└── tests/
    ├── smoke_test.jsonl      # 冒烟测试用例（6 条，无标准答案，用于快速验证流程）
    └── run_<时间戳>.jsonl    # infer.py 输出的预测结果
```

---

## 各组件说明

| 组件 | 职责 |
|---|---|
| `agent/memory.py` | `MemoryManager` — 初始化与更新 Dynamic Memory（Global Query / Task Plan / History Information / Pending Queue） |
| `agent/prompts.py` | 所有 prompt 模板集中管理（中文） |
| `baselines/infer_base.py` | 基线：无搜索直接问答（default）/ 单跳 Jina 搜索（`--jina`） |
| `baselines/infer_react.py` | 基线：Vanilla ReAct，Thought/Action/Observation 循环，仅 Search/Finish |
| `baselines/infer_webdancer.py` | 基线：WebDancer 兼容推理（Serper 搜索 + Jina Reader 访问页面），输出与 eval.py 对齐 |
| `commands/start_judge_vllm.sh` | 启动裁判 vLLM server（端口 6002，默认双卡 TP=2），读取 `judge.model_path` |
| `commands/start_vllm.sh` | 启动推理 vLLM server（端口 6001），支持 `--port`/`-p`、`--gpu`、`--tp` 等参数 |
| `config.yaml` | 统一配置入口：模型路径、vLLM server 端口、Jina API Key、代理、Token 限制、裁判 API |
| `eval.py` | 评分入口：读取预测 JSONL，逐条调用裁判模型，输出 accuracy |
| `infer.py` | 主推理脚本：并发处理 benchmark，每题跑 N 次 rollout，eval.py 报 Pass@N |
| `models/base.py` | `BaseLLM` 抽象类 — 定义 `generate()` 和 `clear_cache()` 接口 |
| `models/vllm_server_model.py` | `VLLMServerModel` — 通过 HTTP API 连接 vLLM server（多线程推荐） |
| `search/jina_search.py` | `JinaSearch` — 调用 Jina Search API，解析 JSON，支持代理 |
| `search_workflow.py` | `SearchWorkflow` — 核心推理循环，orchestrates 多轮搜索 + Dynamic Memory 更新 |
| `tests/smoke_test.jsonl` | 6 条冒烟测试用例，用于快速验证模型能否正常运行 |
| `utils/config_loader.py` | `load_config(path?)` — 加载 YAML，返回 dict |

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
  port: 6001

search:
  jina_api_key: "jina_xxxxxxxxxxxxxxxxxxxx"
  use_proxy: false

eval:
  benchmark_path: "../benchmark/webwalker/main-00000-of-00001.jsonl"
  hotpot_benchmark_path: "../benchmark/hotpot/hotpot_dev_distractor_v1.jsonl"
  output_dir: "tests"

limits:
  max_rounds: 15
  max_new_tokens_default: 1536
  max_final_tokens: 8192
  max_memory_tokens: 1500
  max_filter_tokens: 512
  max_sources_per_search: 5
  max_formatted_sources_len: 4500
```

**评分所需（必填）：**

```yaml
judge:
  model_path: "/path/to/judge/model"
  api_url: "http://127.0.0.1:6002/v1"
  api_key: "EMPTY"
  model: "your-judge-model-name"
```

---

## 推理架构（SearchWorkflow）

`search_workflow.py` 实现核心推理循环，`infer.py` 是并发 runner，每题调用 `SearchWorkflow.run()`。

```
问题 q
│
├─ Round 1: Initial Analysis
│     [Req 1.1] Initial Analysis → 提取 <search> 查询列表
│     → 若无搜索需求（Need Search? No）：进入 Final Round，结束
│     → [Req 1.2] Bootstrap — MemoryManager.initialize() 创建 Dynamic Memory：
│           [Global Query]        ← 根据问题生成研究目标
│           [Task Plan]           ← 生成分步计划
│           [History Information] ← 初始为空
│           [Pending Queue]       ← 填入本轮提取的查询
│     → 进入 Round 2
│
├─ Round 2+: Search Loop（每轮 3 次 LLM 调用）
│   LOOP: Queue Empty OR Max Rounds?
│   │     Yes → 进入 Final Round
│   │     No  ↓
│   ├─ 从 [Pending Queue] 弹出 current_query
│   ├─ [Req 2.1] Relevance Check
│   │     → Relevant? No：跳过，回到循环顶部
│   ├─ [Tool Call] Jina Search → sources
│   ├─ [Req 2.2] Memory Update
│   │     [History Information] += {Query: current_query, Response: 关键事实}
│   │     [Pending Queue]       = remaining_queue + new_queries
│   └─ [Req 2.3] Queue Filtering
│         输入：Dynamic Memory + 全部候选查询
│         输出：过滤后写回 [Pending Queue]（R/W Dynamic Memory）
│         → 回到循环顶部
│
└─ Final Round（1 次 LLM 调用）
      触发条件：Queue Empty 或 Max Rounds 耗尽
      [Req 3.1] Generate Answer
            输入：完整 Dynamic Memory（含全部 [History Information]）+ 原始问题
            输出：最终答案
```

**Dynamic Memory 结构：**

```
[Global Query]
一句话研究目标

[Task Plan]
1. 步骤一
2. 步骤二
...

[History Information]
- Query: <搜索词>
  Response: <核心事实摘要>
- Query: <搜索词>
  Response: <核心事实摘要>

[Pending Queue]
Current queue: query_1, query_2, ...
```

**推理模式对比：**

| 模式 | 脚本 | 搜索 | 结构化记忆 | 多跳规划 | Context 控制 |
|---|---|---|---|---|---|
| 纯 LLM | `baselines/infer_base.py` | ✗ | ✗ | ✗ | — |
| 单跳搜索 | `baselines/infer_base.py --jina` | 1 次 | ✗ | ✗ | — |
| Vanilla ReAct | `baselines/infer_react.py` | 多轮 | ✗ | ✗ | 全历史累积 |
| WebDancer | `baselines/infer_webdancer.py` | 多轮 | ✗ | ✗ | 全历史累积 |
| **Search Agent (Ours)** | `infer.py` | 多轮 | ✓ | ✓ | Dynamic Memory（受控） |

---

## 使用方法

所有命令均在 **`search_agent/` 目录内**运行。

---

### Step 1：启动 vLLM server

```bash
# 单卡，端口 6001
bash commands/start_vllm.sh --gpu 0

# 指定端口和模型路径
bash commands/start_vllm.sh --port 6001 --gpu 0 --model /root/autodl-tmp/Qwen3-8B

# 后台运行
bash commands/start_vllm.sh --port 6001 --gpu 0 -d
```

---

### Step 2：推理

所有推理脚本的 `--benchmark` 参数均接受：`'hotpot'`、`'webwalker'`（从 config.yaml 读路径）或具体文件路径。

#### 主框架（infer.py）—— Memory-Augmented Multi-Round Search

```bash
# 冒烟测试（6 条，快速验证流程）
python infer.py --benchmark tests/smoke_test.jsonl --onetime

# WebWalker benchmark（从 config.yaml 读路径，3 rollout）
python infer.py

# HotpotQA benchmark
python infer.py --benchmark hotpot

# 单次 rollout
python infer.py --onetime

# 自定义 rollout 次数
python infer.py --rollouts 5

# 8 个并发线程
python infer.py --workers 8

# 指定 vLLM server 端口
python infer.py --port 6002

# 只跑前 N 条
python infer.py --limit 10

# 断点续跑（跳过前 N 条）
python infer.py --offset 50 --limit 20

# 完整参数示例
python infer.py \
  --benchmark hotpot \
  --port 6001 \
  --workers 8 \
  --onetime \
  --output tests/hotpot_run.jsonl
```

#### 基线 A（baselines/infer_base.py）—— 直接问答 / 单跳搜索

```bash
# 纯 LLM（无搜索）
python baselines/infer_base.py --port 6001 --onetime

# 单跳 Jina 搜索
python baselines/infer_base.py --port 6001 --onetime --jina
```

#### 基线 B（baselines/infer_react.py）—— Vanilla ReAct

```bash
python baselines/infer_react.py --port 6001 --onetime
python baselines/infer_react.py --port 6001 --onetime --benchmark hotpot
python baselines/infer_react.py --port 6001 --workers 4 --max-rounds 10
```

#### 基线 C（baselines/infer_webdancer.py）—— WebDancer

```bash
# 需要在 config.yaml 中配置 webdancer.serper_api_key
python baselines/infer_webdancer.py --port 6001 --onetime
python baselines/infer_webdancer.py --port 6001 --workers 8 --limit 20
python baselines/infer_webdancer.py --port 6001 --benchmark hotpot --onetime
```

---

### Step 3：评分（eval.py）

```bash
# 1. 启动裁判 vLLM（双卡，端口 6002）
bash commands/start_judge_vllm.sh -d

# 2. 评分
python eval.py --input tests/run_20240101_120000.jsonl

# 调整并发数
python eval.py --input tests/run.jsonl --workers 16
```

**输出示例：**
```
Questions  : 100  ×  3 rollouts
Pass@3     : 58/100  =  58.00%  (≥1 rollout correct)
Avg.Pass   : 48.33%  (145/300 rollouts correct)
Difficulty:
  - easy   : 20/30  =  66.67%
  - medium : 28/50  =  56.00%
  - hard   : 10/20  =  50.00%
```

---

## 文件格式说明

### infer.py 输出（`tests/run_*.jsonl`）

```json
{
  "id": "webwalker_001",
  "question": "美股七姐妹当前市值之和是多少？",
  "gold_answer": "...",
  "type": "factual",
  "level": "hard",
  "root_url": "",
  "info": {"domain": "finance", "difficulty_level": "hard"},
  "rollouts": [
    {
      "rollout_idx": 1,
      "predicted_answer": "七家公司市值之和约为 XX 万亿美元...",
      "used_sources": {"https://...": "Market Cap Data"},
      "final_memory": "[Global Query]\n...\n[History Information]\n...",
      "num_rounds": 8,
      "error": null
    }
  ]
}
```

---

## 扩展指南

### 新增 LLM 后端

1. 在 `models/` 下继承 `BaseLLM`，实现 `generate()` 和 `clear_cache()`
2. 在 `infer.py` 中替换 `VLLMServerModel` 实例化

### 新增搜索后端

1. 在 `search/` 下继承 `BaseSearch`，实现 `search()`，返回 `{"sources": {...}, "error": None}`
2. 在 `search_workflow.py` 中替换 `JinaSearch` 实例化

### 更换裁判模型

修改 `config.yaml` 的 `judge` 块，重启 `commands/start_judge_vllm.sh`，无需改代码。
