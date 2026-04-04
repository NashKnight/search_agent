#!/bin/bash
# start_vllm.sh — Start a vLLM OpenAI-compatible server.
#
# Usage:
#   bash start_vllm.sh [OPTIONS]
#
# Options:
#   --port      PORT     Port to listen on          (default: from config.yaml vllm_server.port, else 6001)
#   --model     PATH     Model path                  (default: from config.yaml model.local_model_path)
#   --gpu       IDS      CUDA_VISIBLE_DEVICES        (default: 0)
#   --tp        N        tensor-parallel-size        (default: 1)
#   --timeout   SECS     Startup wait timeout        (default: 600)
#   --extra     "ARGS"   Extra args passed to vllm   (optional)
#
# Examples:
#   bash start_vllm.sh --port 6001 --gpu 0
#   bash start_vllm.sh --port 6002 --gpu 0,1 --tp 2
#   bash start_vllm.sh --port 6001 --model /data/Qwen3-8B --gpu 0

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/config.yaml"

# ── Defaults ──────────────────────────────────────────────────────────────────
PORT=""
MODEL_PATH=""
GPU="0"
TP=1
TIMEOUT=600
EXTRA_ARGS=""

# ── Parse CLI args ─────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)    PORT="$2";       shift 2 ;;
        --model)   MODEL_PATH="$2"; shift 2 ;;
        --gpu)     GPU="$2";        shift 2 ;;
        --tp)      TP="$2";         shift 2 ;;
        --timeout) TIMEOUT="$2";    shift 2 ;;
        --extra)   EXTRA_ARGS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Read defaults from config.yaml (if not given on CLI) ──────────────────────
if [ -f "$CONFIG_FILE" ]; then
    if [ -z "$PORT" ]; then
        PORT=$(python3 -c "
import yaml, sys
with open('$CONFIG_FILE') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('vllm_server', {}).get('port', 6001))
" 2>/dev/null || echo "6001")
    fi
    if [ -z "$MODEL_PATH" ]; then
        MODEL_PATH=$(python3 -c "
import yaml, sys
with open('$CONFIG_FILE') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('model', {}).get('local_model_path', ''))
" 2>/dev/null || echo "")
    fi
else
    echo "Warning: config.yaml not found at $CONFIG_FILE"
    PORT="${PORT:-6001}"
fi

if [ -z "$MODEL_PATH" ]; then
    echo "Error: model path not set. Use --model or set model.local_model_path in config.yaml"
    exit 1
fi

# ── Launch ────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Model   : $MODEL_PATH"
echo "  GPU(s)  : $GPU"
echo "  Port    : $PORT"
echo "  TP size : $TP"
[ -n "$EXTRA_ARGS" ] && echo "  Extra   : $EXTRA_ARGS"
echo "============================================================"
echo "Starting vLLM server..."

CUDA_VISIBLE_DEVICES="$GPU" vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --disable-log-requests \
    $EXTRA_ARGS &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# ── Wait until ready ──────────────────────────────────────────────────────────
echo "Waiting for server at port $PORT (timeout: ${TIMEOUT}s)..."
START_TIME=$(date +%s)
SPIN=('⠋' '⠙' '⠹' '⠸' '⠼' '⠴' '⠦' '⠧' '⠇' '⠏')
SPIN_IDX=0

while true; do
    if curl -s -f "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; then
        echo ""
        echo "vLLM server is ready on port $PORT!"
        echo "PID $VLLM_PID — press Ctrl+C to stop."
        break
    fi

    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo ""
        echo "Error: vLLM process exited unexpectedly."
        exit 1
    fi

    ELAPSED=$(( $(date +%s) - START_TIME ))
    if [ "$ELAPSED" -gt "$TIMEOUT" ]; then
        echo ""
        echo "Error: Server did not become ready within ${TIMEOUT}s."
        kill "$VLLM_PID" 2>/dev/null || true
        exit 1
    fi

    printf "\r  %s  Elapsed: %ds " "${SPIN[$SPIN_IDX]}" "$ELAPSED"
    SPIN_IDX=$(( (SPIN_IDX + 1) % ${#SPIN[@]} ))
    sleep 2
done

# Keep in foreground — Ctrl+C stops the server
wait "$VLLM_PID"
