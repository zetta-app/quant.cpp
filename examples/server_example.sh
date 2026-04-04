#!/usr/bin/env bash
# server_example.sh -- Start and test the quant.cpp OpenAI-compatible server
#
# Usage:
#   ./examples/server_example.sh <model.gguf>
#
# Prerequisites:
#   cmake -B build -DCMAKE_BUILD_TYPE=Release -DTQ_BUILD_SERVER=ON
#   cmake --build build -j$(nproc)

set -euo pipefail

MODEL="${1:-}"
PORT="${2:-8080}"

if [ -z "$MODEL" ]; then
    echo "Usage: $0 <model.gguf> [port]"
    echo ""
    echo "Example:"
    echo "  $0 Qwen2.5-0.5B-Instruct.gguf 8080"
    exit 1
fi

SERVER="./build/quant-server"
if [ ! -f "$SERVER" ]; then
    echo "Error: quant-server not found. Build with:"
    echo "  cmake -B build -DCMAKE_BUILD_TYPE=Release -DTQ_BUILD_SERVER=ON"
    echo "  cmake --build build -j\$(nproc)"
    exit 1
fi

echo "=== Starting quant.cpp server ==="
echo "Model:  $MODEL"
echo "Port:   $PORT"
echo ""

# Start server in background
$SERVER "$MODEL" -p "$PORT" -j 4 -k uniform_4b &
SERVER_PID=$!

# Wait for server to start
echo "Waiting for server to start..."
for i in $(seq 1 30); do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "Server is ready!"
        break
    fi
    sleep 0.5
done

echo ""
echo "=== Test 1: Health check ==="
curl -s "http://localhost:$PORT/health" | python3 -m json.tool 2>/dev/null || \
    curl -s "http://localhost:$PORT/health"
echo ""

echo ""
echo "=== Test 2: List models ==="
curl -s "http://localhost:$PORT/v1/models" | python3 -m json.tool 2>/dev/null || \
    curl -s "http://localhost:$PORT/v1/models"
echo ""

echo ""
echo "=== Test 3: Chat completion (non-streaming) ==="
curl -s "http://localhost:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "default",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2? Answer in one word."}
        ],
        "max_tokens": 32,
        "temperature": 0.1
    }' | python3 -m json.tool 2>/dev/null || echo "(raw output above)"
echo ""

echo ""
echo "=== Test 4: Chat completion (streaming) ==="
curl -s -N "http://localhost:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "default",
        "messages": [
            {"role": "user", "content": "Say hello in 3 words."}
        ],
        "max_tokens": 32,
        "temperature": 0.7,
        "stream": true
    }'
echo ""

echo ""
echo "=== Test 5: Chat with KV compression options ==="
curl -s "http://localhost:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "default",
        "messages": [
            {"role": "user", "content": "Hello!"}
        ],
        "max_tokens": 16,
        "kv_type": "turbo_kv_3b",
        "value_quant_bits": 4,
        "delta_kv": true
    }' | python3 -m json.tool 2>/dev/null || echo "(raw output above)"
echo ""

echo ""
echo "=== Test 6: OpenAI Python SDK compatibility ==="
cat <<'PYTHON'
# You can also use the official OpenAI Python SDK:
#
#   pip install openai
#
#   from openai import OpenAI
#   client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
#
#   response = client.chat.completions.create(
#       model="default",
#       messages=[{"role": "user", "content": "Hello!"}],
#       max_tokens=64,
#       stream=True,
#   )
#   for chunk in response:
#       if chunk.choices[0].delta.content:
#           print(chunk.choices[0].delta.content, end="", flush=True)
PYTHON

# Cleanup
echo ""
echo "Stopping server (PID $SERVER_PID)..."
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
echo "Done."
