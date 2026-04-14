#!/bin/bash
# start_judge_vllm.sh — Start the judge vLLM server (2-GPU, port 6002).
#
# Usage:
#   bash commands/start_judge_vllm.sh [OPTIONS]
#   bash commands/start_judge_vllm.sh --stop          # stop background judge server
#   bash commands/start_judge_vllm.sh --logs          # tail judge server logs
#
# Options:
#   --port      PORT   Port to listen on         (default: 6002)
#   --model/-m  PATH   Model path                (default: from config.yaml judge.model_path)
#   --gpu/-g    IDS    CUDA_VISIBLE_DEVICES      (default: 0,1)
#   --tp        N      tensor-parallel-size      (default: 2)
#   --timeout   SECS   Startup wait timeout      (default: 600, foreground only)
#   --max-model-len N  Max sequence length       (default: from config.yaml judge.max_model_len)
#   --gpu-mem-util  F  GPU memory utilization    (default: from config.yaml judge.gpu_memory_utilization)
#   --extra     "ARGS" Extra args passed to vllm (optional)
#   -d                 Detach: run in background, return immediately

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$PROJECT_ROOT/config.yaml"
PID_FILE="$PROJECT_ROOT/judge_vllm.pid"
LOG_FILE="$PROJECT_ROOT/logs/start_judge_vllm_$(date +%Y%m%d_%H%M%S).log"

# ── Stop / logs shortcuts ──────────────────────────────────────────────────────
mkdir -p "$PROJECT_ROOT/logs"

if [[ "${1:-}" == "--stop" ]]; then
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID"
            rm -f "$PID_FILE"
            echo "Judge vLLM server (PID $PID) stopped."
        else
            echo "No running process found for PID $PID. Removing stale PID file."
            rm -f "$PID_FILE"
        fi
    else
        echo "No PID file found. Is the judge server running?"
    fi
    exit 0
fi

if [[ "${1:-}" == "--logs" ]]; then
    LATEST_LOG=$(ls -t "$PROJECT_ROOT"/logs/start_judge_vllm_*.log 2>/dev/null | head -1)
    [ -z "$LATEST_LOG" ] && echo "No log files found." && exit 1
    echo "Tailing: $LATEST_LOG"
    tail -f "$LATEST_LOG"
    exit 0
fi

# ── Defaults ──────────────────────────────────────────────────────────────────
PORT="6002"
MODEL_PATH=""
GPU="0,1"
TP=2
TIMEOUT=600
EXTRA_ARGS=""
MAX_MODEL_LEN=""
GPU_MEM_UTIL=""
DETACH=false

# ── Parse CLI args ─────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port|-p)       PORT="$2";          shift 2 ;;
        --model|-m)      MODEL_PATH="$2";    shift 2 ;;
        --gpu|-g)        GPU="$2";           shift 2 ;;
        --tp)            TP="$2";            shift 2 ;;
        --timeout)       TIMEOUT="$2";       shift 2 ;;
        --extra)         EXTRA_ARGS="$2";    shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        --gpu-mem-util)  GPU_MEM_UTIL="$2";  shift 2 ;;
        -d)              DETACH=true;        shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Read defaults from config.yaml ────────────────────────────────────────────
if [ -f "$CONFIG_FILE" ]; then
    if [ -z "$MODEL_PATH" ]; then
        MODEL_PATH=$(python3 -c "
import yaml
with open('$CONFIG_FILE') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('judge', {}).get('model_path', ''))
" 2>/dev/null || echo "")
    fi
    if [ -z "$MAX_MODEL_LEN" ]; then
        MAX_MODEL_LEN=$(python3 -c "
import yaml
with open('$CONFIG_FILE') as f:
    cfg = yaml.safe_load(f)
v = cfg.get('judge', {}).get('max_model_len', '')
print(v if v else '')
" 2>/dev/null || echo "")
    fi
    if [ -z "$GPU_MEM_UTIL" ]; then
        GPU_MEM_UTIL=$(python3 -c "
import yaml
with open('$CONFIG_FILE') as f:
    cfg = yaml.safe_load(f)
v = cfg.get('judge', {}).get('gpu_memory_utilization', '')
print(v if v else '')
" 2>/dev/null || echo "")
    fi
else
    echo "Warning: config.yaml not found at $CONFIG_FILE"
fi

if [ -z "$MODEL_PATH" ]; then
    echo "Error: judge model path not set. Use --model/-m or set judge.model_path in config.yaml"
    exit 1
fi

# ── Check if already running ──────────────────────────────────────────────────
if $DETACH && [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Judge vLLM is already running (PID $OLD_PID). Use --stop to stop it first."
        exit 1
    else
        rm -f "$PID_FILE"
    fi
fi

# ── Print config ───────────────────────────────────────────────────────────────
echo "============================================================"
echo "  [Judge Server]"
echo "  Model   : $MODEL_PATH"
echo "  GPU(s)  : $GPU"
echo "  Port    : $PORT"
echo "  TP size : $TP"
[ -n "$MAX_MODEL_LEN" ] && echo "  Max len : $MAX_MODEL_LEN"
[ -n "$GPU_MEM_UTIL" ]  && echo "  GPU mem : $GPU_MEM_UTIL"
[ -n "$EXTRA_ARGS" ]    && echo "  Extra   : $EXTRA_ARGS"
$DETACH && echo "  Mode    : background (nohup)" || echo "  Mode    : foreground"
echo "============================================================"

VLLM_CMD=(
    vllm serve "$MODEL_PATH"
    --host 0.0.0.0
    --port "$PORT"
    --tensor-parallel-size "$TP"
    --no-enable-log-requests
    ${MAX_MODEL_LEN:+--max-model-len $MAX_MODEL_LEN}
    ${GPU_MEM_UTIL:+--gpu-memory-utilization $GPU_MEM_UTIL}
)
[ -n "$EXTRA_ARGS" ] && VLLM_CMD+=($EXTRA_ARGS)

# ── Launch ────────────────────────────────────────────────────────────────────
if $DETACH; then
    CUDA_VISIBLE_DEVICES="$GPU" nohup "${VLLM_CMD[@]}" > "$LOG_FILE" 2>&1 &
    VLLM_PID=$!
    echo "$VLLM_PID" > "$PID_FILE"
    echo "Judge vLLM started in background (PID $VLLM_PID)"
    echo ""
    echo "  Stop : bash commands/start_judge_vllm.sh --stop"
    echo "  Logs : bash commands/start_judge_vllm.sh --logs"
else
    echo "Starting judge vLLM server (Ctrl+C to stop)..."
    START_TIME=$(date +%s)
    SPIN=('⠋' '⠙' '⠹' '⠸' '⠼' '⠴' '⠦' '⠧' '⠇' '⠏')
    SPIN_IDX=0

    CUDA_VISIBLE_DEVICES="$GPU" "${VLLM_CMD[@]}" &
    VLLM_PID=$!

    while true; do
        if curl -s -f "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; then
            echo ""
            echo "Judge vLLM server is ready on port $PORT! (Ctrl+C to stop)"
            wait "$VLLM_PID"
            break
        fi

        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo ""
            echo "Error: Judge vLLM process exited unexpectedly."
            exit 1
        fi

        ELAPSED=$(( $(date +%s) - START_TIME ))
        if [ "$ELAPSED" -gt "$TIMEOUT" ]; then
            echo ""
            echo "Error: Judge server did not become ready within ${TIMEOUT}s."
            kill "$VLLM_PID" 2>/dev/null || true
            exit 1
        fi

        printf "\r  %s  Elapsed: %ds " "${SPIN[$SPIN_IDX]}" "$ELAPSED"
        SPIN_IDX=$(( (SPIN_IDX + 1) % ${#SPIN[@]} ))
        sleep 2
    done
fi
