# Search Agent 6.0

基于 vLLM + Jina Search 构建的多轮搜索智能体。

**v6.0（当前版本）**：`infer.py` 升级为 **MA-HReAct**（Memory-Augmented Hierarchical ReAct）架构。引入三阶段多调用设计（Decomposition → Execution Loop → Synthesis），通过结构化工作记忆（SWM）和显式查询队列实现多跳推理，同时保持每条问题的 context 受控（每次 LLM 调用只看当前 memory + 当前 observation，不传递完整对话历史）。

**v5.1 特性保留**：默认每题跑 3 次 rollout，eval.py 报 Pass@3 和 Avg.Pass，与 WebDancer 评测体系对齐。

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
│   ├── memory.py             # MemoryManager（v5.x prototype 使用，v6.0 已内联）
│   ├── prompts.py            # v5.x prompt 模板集中管理
│   └── __init__.py
│
├── search_workflow.py        # v5.x SearchWorkflow 主循环（prototype）
├── infer.py                  # 主推理脚本：MA-HReAct 三阶段架构（v6.0）
├── infer_prototype.py        # v5.x prototype 推理脚本（memory board + 队列过滤）
├── infer_react.py            # 基线推理：Vanilla ReAct（Thought/Action/Observation 循环）
├── infer_base.py             # 基线推理：纯 LLM 直接问答 / 单跳 Jina 搜索（--jina）
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
| `start_vllm.sh` | 启动推理 vLLM server（端口 6001），支持 `--port`/`-p`、`--gpu`、`--tp` 等参数 |
| `start_judge_vllm.sh` | 启动裁判 vLLM server（端口 6002，默认双卡 TP=2），读取 `judge.model_path` |
| `utils/config_loader.py` | `load_config(path?)` — 加载 YAML，返回 dict |
| `models/base.py` | `BaseLLM` 抽象类 — 定义 `generate()` 和 `clear_cache()` 接口 |
| `models/vllm_server_model.py` | `VLLMServerModel` — 通过 HTTP API 连接 vLLM server（多线程推荐） |
| `search/jina_search.py` | `JinaSearch` — 调用 Jina Search API，解析 JSON，支持代理 |
| `infer.py` | **主推理脚本（v6.0）**：MA-HReAct 三阶段多调用架构，支持多跳推理 |
| `infer_prototype.py` | v5.x prototype：memory board + 队列过滤，多轮 LLM 调用 |
| `infer_react.py` | 基线：Vanilla ReAct，Thought/Action/Observation 循环，仅 Search/Finish |
| `infer_base.py` | 基线：无搜索直接问答（default）/ 单跳 Jina 搜索（`--jina`） |
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
  max_final_tokens: 4096
  max_filter_tokens: 512
  max_sources_per_search: 5
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

## infer.py（v6.0）推理架构详解

`infer.py` 使用 MA-HReAct（Memory-Augmented Hierarchical ReAct）架构，每个问题的推理流程分为三个阶段，每个 LLM 调用只有单一职责。

```
问题 q
│
├─ Phase 1: Decomposition（1 次 LLM 调用）
│     输入：问题文本
│     输出：<goal> + <plan> + 初始 <tool_call> 列表
│     → 初始化 SWM（goal, plan, 空 history, 初始 pending queue）
│     → 若模型输出 <finish>，直接返回答案（无需搜索）
│
├─ Phase 2: Execution Loop（每轮 2 次 LLM 调用）
│   LOOP: pending_queue 非空 且 round <= max_rounds
│   │
│   ├─ 从 pending_queue 弹出 current_query
│   ├─ JinaSearch 执行搜索 → observation
│   │
│   ├─ Request 2.1: Analysis + Compression（1 次 LLM 调用）
│   │     输入：SWM + observation
│   │     输出（必须）：<new_compressed_history>
│   │     输出（可选）：<tool_call> × N  或  <finish>
│   │     → 将 key_fact 写入 SWM.compressed_history
│   │     → 若 <finish>：返回答案，结束
│   │     → 将新 <tool_call> 加入 pending_queue（精确去重）
│   │
│   └─ Request 2.2: Queue Filter（1 次 LLM 调用）
│         输入：更新后的 SWM（含新 history + 新 pending_queue）
│         输出：<keep> / <remove> 决策
│         → 剔除已搜索过的、无关的、重复的查询
│         → 若全部删除 / <queue_empty/>：进入 Phase 3
│
└─ Phase 3: Synthesis（1 次 LLM 调用）
      触发条件：pending_queue 为空 或 max_rounds 耗尽
      输入：完整 SWM（含所有 compressed_history）+ 原始问题
      输出：最终答案
```

**SWM（Structured Working Memory）结构：**

```
[Goal]
一句话研究目标

[Research Plan]
1. 步骤一
2. 步骤二
...

[Search History]
- Q: query1 | A: key facts in 1-2 sentences
- Q: query2 | A: key facts in 1-2 sentences
...

[Pending Queue]
- pending_query_1
- pending_query_2
```

**关键数据流：**

```
question
  → [LLM] Decomposition → goal + plan + tool_calls
       → SWM 初始化，队列填充
            → LOOP: pop → Jina search → [LLM] Analysis(obs+SWM) → history + new queries
                        → [LLM] Filter(SWM) → 剪枝队列
                              → 队列空 or max_rounds → [LLM] Synthesis(SWM) → answer
```

---

## 使用方法

所有命令均在 **`search_agent/` 目录内**运行。

---

### Step 1：启动 vLLM server

```bash
# 单卡，端口 6001
bash start_vllm.sh --gpu 0

# 指定端口和模型路径
bash start_vllm.sh --port 6001 --gpu 0 --model /root/autodl-tmp/Qwen3-8B

# 后台运行
nohup bash start_vllm.sh --port 6001 --gpu 0 > vllm_6001.log 2>&1 &
```

---

### Step 2：推理

#### 主框架（infer.py）—— MA-HReAct 三阶段架构

```bash
# 冒烟测试（6 条，快速验证流程）
python infer.py --benchmark tests/smoke_test.jsonl --onetime

# WebWalker benchmark（从 config.yaml 读路径，3 rollout）
python infer.py

# HotpotQA benchmark（多跳推理专项测试）
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

#### 基线 A（infer_react.py）—— Vanilla ReAct

```bash
python infer_react.py --port 6001 --onetime
python infer_react.py --port 6001 --onetime --benchmark hotpot
python infer_react.py --port 6001 --workers 4 --max-rounds 10
```

#### 基线 B（infer_base.py）—— 直接问答 / 单跳搜索

```bash
# 纯 LLM（无搜索）
python infer_base.py --port 6001 --onetime

# 单跳 Jina 搜索
python infer_base.py --port 6001 --onetime --jina
```

#### 基线 C（infer_prototype.py）—— v5.x prototype

```bash
python infer_prototype.py --port 6001 --onetime
python infer_prototype.py --port 6001 --workers 4
```

**推理模式对比：**

| 模式 | 脚本 | 搜索 | 结构化记忆 | 多跳规划 | Context 控制 |
|---|---|---|---|---|---|
| 纯 LLM | `infer_base.py` | ✗ | ✗ | ✗ | — |
| 单跳搜索 | `infer_base.py --jina` | 1 次 | ✗ | ✗ | — |
| Vanilla ReAct | `infer_react.py` | 多轮 | ✗ | ✗ | 全历史累积 |
| Prototype (v5.x) | `infer_prototype.py` | 多轮 | ✓ | 部分 | 全历史累积 |
| **MA-HReAct (v6.0)** | `infer.py` | 多轮 | ✓ | ✓ | 受控（仅当前轮） |

---

### Step 3：评分（eval.py）

```bash
# 1. 启动裁判 vLLM（双卡，端口 6002）
bash start_judge_vllm.sh -d

# 2. 评分
python eval.py --input tests/run_mahreact_20240101_120000.jsonl

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
  "question": "美股七姐妹当前市值之和是多少？",
  "gold_answer": "...",
  "root_url": "",
  "info": {"domain": "finance", "difficulty_level": "hard"},
  "rollouts": [
    {
      "rollout_idx": 1,
      "predicted_answer": "七家公司市值之和约为 XX 万亿美元...",
      "used_sources": {"https://...": "Market Cap Data"},
      "final_memory": "[Goal]\n...\n[Search History]\n- Q: ... | A: ...",
      "num_rounds": 8,
      "error": null
    }
  ]
}
```

### config.yaml 新增字段（v6.0）

```yaml
eval:
  hotpot_benchmark_path: "../benchmark/hotpot/hotpot_dev_distractor_v1.jsonl"

limits:
  max_final_tokens: 4096    # synthesis 阶段 max tokens
  max_filter_tokens: 512    # filter 阶段 max tokens
```

---

## 扩展指南

### 新增 LLM 后端

1. 在 `models/` 下继承 `BaseLLM`，实现 `generate()` 和 `clear_cache()`
2. 在 `infer.py` 中替换 `VLLMServerModel` 实例化

### 新增搜索后端

1. 在 `search/` 下继承 `BaseSearch`，实现 `search()`，返回 `{"sources": {...}, "error": None}`
2. 在 `infer.py` 中替换 `JinaSearch` 实例化

### 更换裁判模型

修改 `config.yaml` 的 `judge` 块，重启 `start_judge_vllm.sh`，无需改代码。
