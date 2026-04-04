#!/usr/bin/env bash
# =============================================================================
# long_context_ppl_bench.sh — Long-Context Perplexity Benchmark
# =============================================================================
#
# Measures perplexity at 1K, 4K, 8K, 16K token context lengths, comparing
# FP16 baseline, 4-bit uniform KV, and delta 3-bit KV compression.
#
# This answers the community question: "Does KV compression quality hold
# over long sequences, not just the 100-999 token range?"
#
# Usage:
#   bash bench/long_context_ppl_bench.sh <model.gguf> [threads]
#
# Example:
#   bash bench/long_context_ppl_bench.sh models/Llama-3.2-1B-Q8_0.gguf 8
#
# Prerequisites:
#   1. Build quant: cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)
#   2. Generate test data: python3 bench/generate_long_text.py
#      (or provide your own text files in bench/data/ppl_{1k,4k,8k,16k}.txt)
#
# Output:
#   - Results table on stdout
#   - CSV at bench/data/long_context_ppl_results.csv
#   - JSON per-run at bench/data/long_context_ppl_*.json
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_DIR/build"
TQ_RUN="$BUILD_DIR/quant"
DATA_DIR="$SCRIPT_DIR/data"

MODEL="${1:?Usage: bash bench/long_context_ppl_bench.sh <model.gguf> [threads]}"
THREADS="${2:-$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)}"

# Context length targets and corresponding data files
# Format: "label tokens filename ctx_override"
CONTEXT_CONFIGS=(
    "1K    1024   ppl_1k.txt    1024"
    "4K    4096   ppl_4k.txt    4096"
    "8K    8192   ppl_8k.txt    8192"
    "16K   16384  ppl_16k.txt   16384"
)

# KV configurations to benchmark
# Format: "label kv_flag v_flag delta_flag description"
KV_CONFIGS=(
    "FP16_baseline     fp32        fp16  0  FP16 K + FP16 V (no compression)"
    "uniform_4b+Q4V    uniform_4b  q4    0  4-bit uniform K + Q4 V"
    "delta_3b+Q4V      turbo_kv_3b q4    1  delta 3-bit K + Q4 V"
    "turbo_3b+Q4V      turbo_kv_3b q4    0  turbo 3-bit K + Q4 V (no delta)"
    "turbo_1b+Q4V      turbo_kv_1b q4    0  turbo 1-bit K + Q4 V"
)

# Weight quantization (Q4 for all runs to keep weight quality constant)
WEIGHT_QUANT="q4"

DATE_STR=$(date +%Y-%m-%d_%H%M%S)
CSV_OUT="$DATA_DIR/long_context_ppl_results.csv"
JSON_DIR="$DATA_DIR/long_context_ppl_json"

# --------------------------------------------------------
# Validate prerequisites
# --------------------------------------------------------

if [ ! -f "$TQ_RUN" ]; then
    echo "ERROR: $TQ_RUN not found."
    echo "Build first:"
    echo "  cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j\$(nproc)"
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model not found: $MODEL"
    exit 1
fi

# Check for test data files; generate if missing
MISSING_DATA=0
for config_line in "${CONTEXT_CONFIGS[@]}"; do
    filename=$(echo "$config_line" | awk '{print $3}')
    if [ ! -f "$DATA_DIR/$filename" ]; then
        MISSING_DATA=1
        break
    fi
done

if [ "$MISSING_DATA" -eq 1 ]; then
    echo "Some test data files are missing. Generating..."
    echo ""
    if command -v python3 &>/dev/null; then
        python3 "$SCRIPT_DIR/generate_long_text.py"
        echo ""
    else
        echo "ERROR: python3 not found. Please generate test data manually:"
        echo "  python3 bench/generate_long_text.py"
        exit 1
    fi
fi

# Verify at least 1K file exists (minimum)
if [ ! -f "$DATA_DIR/ppl_1k.txt" ]; then
    echo "ERROR: bench/data/ppl_1k.txt not found even after generation."
    echo "Please provide text files manually or check generate_long_text.py output."
    exit 1
fi

mkdir -p "$JSON_DIR"

MODEL_NAME=$(basename "$MODEL")

# --------------------------------------------------------
# Print header
# --------------------------------------------------------
echo "============================================================"
echo "  quant.cpp Long-Context Perplexity Benchmark"
echo "============================================================"
echo ""
echo "  Model:     $MODEL_NAME"
echo "  Weights:   $WEIGHT_QUANT"
echo "  Threads:   $THREADS"
echo "  Date:      $DATE_STR"
echo ""
echo "  Context lengths: 1K, 4K, 8K, 16K tokens"
echo "  KV configs:"
for kv_line in "${KV_CONFIGS[@]}"; do
    label=$(echo "$kv_line" | awk '{print $1}')
    desc=$(echo "$kv_line" | awk '{for(i=5;i<=NF;i++) printf "%s ", $i; print ""}' | sed 's/ *$//')
    printf "    %-20s %s\n" "$label" "$desc"
done
echo ""
echo "============================================================"
echo ""

# --------------------------------------------------------
# CSV header
# --------------------------------------------------------
echo "date,model,context_label,context_tokens,kv_label,kv_type,v_quant,delta,tokens_eval,nll,ppl,tok_s" > "$CSV_OUT"

# --------------------------------------------------------
# Declare arrays to store results for the summary table
# --------------------------------------------------------
# We store results as "ctx_label|kv_label|ppl|nll|tok_s|tokens_eval"
declare -a RESULTS=()

# --------------------------------------------------------
# Run benchmarks
# --------------------------------------------------------
TOTAL_RUNS=0
COMPLETED_RUNS=0

# Count total runs
for ctx_line in "${CONTEXT_CONFIGS[@]}"; do
    filename=$(echo "$ctx_line" | awk '{print $3}')
    if [ -f "$DATA_DIR/$filename" ]; then
        for kv_line in "${KV_CONFIGS[@]}"; do
            TOTAL_RUNS=$((TOTAL_RUNS + 1))
        done
    fi
done

echo "Running $TOTAL_RUNS benchmark configurations..."
echo "(This may take a while for 8K/16K contexts)"
echo ""

for ctx_line in "${CONTEXT_CONFIGS[@]}"; do
    ctx_label=$(echo "$ctx_line" | awk '{print $1}')
    ctx_tokens=$(echo "$ctx_line" | awk '{print $2}')
    filename=$(echo "$ctx_line" | awk '{print $3}')
    ctx_override=$(echo "$ctx_line" | awk '{print $4}')

    filepath="$DATA_DIR/$filename"

    if [ ! -f "$filepath" ]; then
        echo "  SKIP: $filename not found (source text too short for $ctx_label)"
        for kv_line in "${KV_CONFIGS[@]}"; do
            kv_label=$(echo "$kv_line" | awk '{print $1}')
            RESULTS+=("${ctx_label}|${kv_label}|N/A|N/A|N/A|0")
        done
        continue
    fi

    word_count=$(wc -w < "$filepath" | tr -d ' ')
    echo "--- Context: $ctx_label ($word_count words from $filename) ---"

    for kv_line in "${KV_CONFIGS[@]}"; do
        kv_label=$(echo "$kv_line" | awk '{print $1}')
        kv_type=$(echo "$kv_line" | awk '{print $2}')
        v_quant=$(echo "$kv_line" | awk '{print $3}')
        delta_flag=$(echo "$kv_line" | awk '{print $4}')

        COMPLETED_RUNS=$((COMPLETED_RUNS + 1))
        printf "  [%d/%d] %-20s ... " "$COMPLETED_RUNS" "$TOTAL_RUNS" "$kv_label"

        # Build command
        CMD="$TQ_RUN $MODEL --ppl $filepath -j $THREADS -k $kv_type -q $WEIGHT_QUANT --ctx $ctx_override --json"
        if [ "$v_quant" != "fp16" ]; then
            CMD="$CMD -v $v_quant"
        fi
        if [ "$delta_flag" = "1" ]; then
            CMD="$CMD --delta"
        fi

        # Run and capture output (JSON on stdout, progress on stderr)
        json_file="$JSON_DIR/${DATE_STR}_${ctx_label}_${kv_label}.json"
        stderr_out=""
        if json_out=$($CMD 2>"${json_file}.stderr"); then
            stderr_out=$(cat "${json_file}.stderr")

            # Save JSON
            echo "$json_out" > "$json_file"

            # Parse PPL_CSV from stderr: PPL_CSV:<tokens>,<nll>,<ppl>
            ppl_csv_line=$(echo "$stderr_out" | grep "^PPL_CSV:" || true)
            if [ -n "$ppl_csv_line" ]; then
                tokens_eval=$(echo "$ppl_csv_line" | sed 's/PPL_CSV://' | cut -d, -f1)
                nll=$(echo "$ppl_csv_line" | sed 's/PPL_CSV://' | cut -d, -f2)
                ppl=$(echo "$ppl_csv_line" | sed 's/PPL_CSV://' | cut -d, -f3)
            else
                # Fallback: parse from JSON output
                tokens_eval=$(echo "$json_out" | grep '"tokens_evaluated"' | grep -o '[0-9]*' || echo "0")
                ppl=$(echo "$json_out" | grep '"perplexity"' | grep -o '[0-9]*\.[0-9]*' || echo "N/A")
                nll=$(echo "$json_out" | grep '"avg_nll"' | grep -o '[0-9]*\.[0-9]*' || echo "N/A")
            fi

            # Parse tok/s from stderr
            tok_s=$(echo "$stderr_out" | grep "tok/s" | tail -1 | grep -o '[0-9]*\.[0-9]* tok/s' | grep -o '[0-9]*\.[0-9]*' || echo "N/A")
            if [ -z "$tok_s" ] || [ "$tok_s" = "" ]; then
                tok_s=$(echo "$json_out" | grep '"tok_per_s"' | grep -o '[0-9]*\.[0-9]*' || echo "N/A")
            fi

            printf "PPL=%-8s NLL=%-8s (%s tok/s, %s tokens)\n" "$ppl" "$nll" "$tok_s" "$tokens_eval"
        else
            stderr_out=$(cat "${json_file}.stderr" 2>/dev/null || true)
            tokens_eval="0"
            nll="ERR"
            ppl="ERR"
            tok_s="N/A"

            # Check if it was a context-too-short issue
            if echo "$stderr_out" | grep -qi "need at least 2 tokens"; then
                ppl="TOO_SHORT"
                printf "SKIP (text too short for tokenizer)\n"
            elif echo "$stderr_out" | grep -qi "out of memory\|alloc"; then
                ppl="OOM"
                printf "SKIP (out of memory at $ctx_label context)\n"
            else
                printf "FAILED\n"
                # Show first line of error for debugging
                echo "$stderr_out" | head -3 | sed 's/^/    /' >&2
            fi
        fi

        # Clean up stderr temp file
        rm -f "${json_file}.stderr"

        # Store result
        RESULTS+=("${ctx_label}|${kv_label}|${ppl}|${nll}|${tok_s}|${tokens_eval}")

        # Write CSV row
        echo "$DATE_STR,$MODEL_NAME,$ctx_label,$ctx_tokens,$kv_label,$kv_type,$v_quant,$delta_flag,$tokens_eval,$nll,$ppl,$tok_s" >> "$CSV_OUT"
    done
    echo ""
done

# --------------------------------------------------------
# Summary Table: PPL by context length and KV config
# --------------------------------------------------------
echo ""
echo "============================================================"
echo "  RESULTS: Perplexity by Context Length"
echo "============================================================"
echo ""

# Collect unique context labels and KV labels (preserving order)
CTX_LABELS=()
for ctx_line in "${CONTEXT_CONFIGS[@]}"; do
    CTX_LABELS+=("$(echo "$ctx_line" | awk '{print $1}')")
done

KV_LABELS=()
for kv_line in "${KV_CONFIGS[@]}"; do
    KV_LABELS+=("$(echo "$kv_line" | awk '{print $1}')")
done

# Print table header
printf "  %-22s" "KV Config"
for ctx in "${CTX_LABELS[@]}"; do
    printf "  %10s" "$ctx"
done
echo ""

printf "  %-22s" "----------------------"
for ctx in "${CTX_LABELS[@]}"; do
    printf "  %10s" "----------"
done
echo ""

# Print rows
for kv in "${KV_LABELS[@]}"; do
    printf "  %-22s" "$kv"
    for ctx in "${CTX_LABELS[@]}"; do
        ppl="--"
        for result in "${RESULTS[@]}"; do
            r_ctx=$(echo "$result" | cut -d'|' -f1)
            r_kv=$(echo "$result" | cut -d'|' -f2)
            r_ppl=$(echo "$result" | cut -d'|' -f3)
            if [ "$r_ctx" = "$ctx" ] && [ "$r_kv" = "$kv" ]; then
                ppl="$r_ppl"
                break
            fi
        done
        printf "  %10s" "$ppl"
    done
    echo ""
done

# --------------------------------------------------------
# Degradation Table: % change vs FP16 baseline
# --------------------------------------------------------
echo ""
echo "============================================================"
echo "  DEGRADATION: % PPL increase vs FP16 baseline"
echo "============================================================"
echo ""

# Get baseline label (first KV config)
BASELINE_LABEL="${KV_LABELS[0]}"

printf "  %-22s" "KV Config"
for ctx in "${CTX_LABELS[@]}"; do
    printf "  %10s" "$ctx"
done
echo ""

printf "  %-22s" "----------------------"
for ctx in "${CTX_LABELS[@]}"; do
    printf "  %10s" "----------"
done
echo ""

for kv in "${KV_LABELS[@]}"; do
    printf "  %-22s" "$kv"
    for ctx in "${CTX_LABELS[@]}"; do
        # Find baseline PPL for this context
        base_ppl=""
        for result in "${RESULTS[@]}"; do
            r_ctx=$(echo "$result" | cut -d'|' -f1)
            r_kv=$(echo "$result" | cut -d'|' -f2)
            r_ppl=$(echo "$result" | cut -d'|' -f3)
            if [ "$r_ctx" = "$ctx" ] && [ "$r_kv" = "$BASELINE_LABEL" ]; then
                base_ppl="$r_ppl"
                break
            fi
        done

        # Find this config's PPL
        this_ppl=""
        for result in "${RESULTS[@]}"; do
            r_ctx=$(echo "$result" | cut -d'|' -f1)
            r_kv=$(echo "$result" | cut -d'|' -f2)
            r_ppl=$(echo "$result" | cut -d'|' -f3)
            if [ "$r_ctx" = "$ctx" ] && [ "$r_kv" = "$kv" ]; then
                this_ppl="$r_ppl"
                break
            fi
        done

        # Calculate % degradation
        if [ "$kv" = "$BASELINE_LABEL" ]; then
            printf "  %10s" "(base)"
        elif [ -z "$base_ppl" ] || [ -z "$this_ppl" ] || \
             [ "$base_ppl" = "N/A" ] || [ "$this_ppl" = "N/A" ] || \
             [ "$base_ppl" = "ERR" ] || [ "$this_ppl" = "ERR" ] || \
             [ "$base_ppl" = "TOO_SHORT" ] || [ "$this_ppl" = "TOO_SHORT" ] || \
             [ "$base_ppl" = "OOM" ] || [ "$this_ppl" = "OOM" ]; then
            printf "  %10s" "--"
        else
            pct=$(echo "scale=2; ($this_ppl - $base_ppl) / $base_ppl * 100" | bc 2>/dev/null || echo "N/A")
            if [ "$pct" != "N/A" ]; then
                # Add + prefix for positive values
                if echo "$pct" | grep -q '^-'; then
                    printf "  %9s%%" "$pct"
                else
                    printf "  %8s%%" "+$pct"
                fi
            else
                printf "  %10s" "--"
            fi
        fi
    done
    echo ""
done

# --------------------------------------------------------
# Speed Table: tok/s by context length
# --------------------------------------------------------
echo ""
echo "============================================================"
echo "  SPEED: Tokens/second by Context Length"
echo "============================================================"
echo ""

printf "  %-22s" "KV Config"
for ctx in "${CTX_LABELS[@]}"; do
    printf "  %10s" "$ctx"
done
echo ""

printf "  %-22s" "----------------------"
for ctx in "${CTX_LABELS[@]}"; do
    printf "  %10s" "----------"
done
echo ""

for kv in "${KV_LABELS[@]}"; do
    printf "  %-22s" "$kv"
    for ctx in "${CTX_LABELS[@]}"; do
        tok_s="--"
        for result in "${RESULTS[@]}"; do
            r_ctx=$(echo "$result" | cut -d'|' -f1)
            r_kv=$(echo "$result" | cut -d'|' -f2)
            r_tok_s=$(echo "$result" | cut -d'|' -f5)
            if [ "$r_ctx" = "$ctx" ] && [ "$r_kv" = "$kv" ]; then
                tok_s="$r_tok_s"
                break
            fi
        done
        printf "  %10s" "$tok_s"
    done
    echo ""
done

# --------------------------------------------------------
# Key findings
# --------------------------------------------------------
echo ""
echo "============================================================"
echo "  ANALYSIS"
echo "============================================================"
echo ""

# Check if PPL degradation stays under threshold across lengths
THRESHOLD=5.0  # 5% PPL degradation threshold
GOOD_CONFIGS=0
TESTED_CONFIGS=0

for kv in "${KV_LABELS[@]}"; do
    [ "$kv" = "$BASELINE_LABEL" ] && continue
    worst_pct=0
    has_data=0

    for ctx in "${CTX_LABELS[@]}"; do
        base_ppl=""
        this_ppl=""
        for result in "${RESULTS[@]}"; do
            r_ctx=$(echo "$result" | cut -d'|' -f1)
            r_kv=$(echo "$result" | cut -d'|' -f2)
            r_ppl=$(echo "$result" | cut -d'|' -f3)
            if [ "$r_ctx" = "$ctx" ] && [ "$r_kv" = "$BASELINE_LABEL" ]; then base_ppl="$r_ppl"; fi
            if [ "$r_ctx" = "$ctx" ] && [ "$r_kv" = "$kv" ]; then this_ppl="$r_ppl"; fi
        done

        if [ -n "$base_ppl" ] && [ -n "$this_ppl" ] && \
           [ "$base_ppl" != "N/A" ] && [ "$this_ppl" != "N/A" ] && \
           [ "$base_ppl" != "ERR" ] && [ "$this_ppl" != "ERR" ]; then
            pct=$(echo "scale=4; ($this_ppl - $base_ppl) / $base_ppl * 100" | bc 2>/dev/null || echo "0")
            has_data=1
            # Track worst degradation (use absolute comparison with bc)
            is_worse=$(echo "$pct > $worst_pct" | bc 2>/dev/null || echo "0")
            if [ "$is_worse" = "1" ]; then
                worst_pct="$pct"
            fi
        fi
    done

    if [ "$has_data" = "1" ]; then
        TESTED_CONFIGS=$((TESTED_CONFIGS + 1))
        is_good=$(echo "$worst_pct < $THRESHOLD" | bc 2>/dev/null || echo "0")
        if [ "$is_good" = "1" ]; then
            GOOD_CONFIGS=$((GOOD_CONFIGS + 1))
            pct_fmt=$(printf "%.1f" "$worst_pct" 2>/dev/null || echo "$worst_pct")
            echo "  [PASS] $kv: worst degradation ${pct_fmt}% (< ${THRESHOLD}% threshold)"
        else
            pct_fmt=$(printf "%.1f" "$worst_pct" 2>/dev/null || echo "$worst_pct")
            echo "  [WARN] $kv: worst degradation ${pct_fmt}% (>= ${THRESHOLD}% threshold)"
        fi
    fi
done

if [ "$TESTED_CONFIGS" -gt 0 ]; then
    echo ""
    echo "  $GOOD_CONFIGS/$TESTED_CONFIGS configs passed the <${THRESHOLD}% PPL degradation threshold."
fi

echo ""
echo "============================================================"
echo "  Files saved:"
echo "    CSV:  $CSV_OUT"
echo "    JSON: $JSON_DIR/"
echo "============================================================"
echo ""
echo "  To re-run a single configuration:"
echo "    $TQ_RUN $MODEL --ppl bench/data/ppl_4k.txt -k turbo_kv_3b -v q4 --delta -q $WEIGHT_QUANT --ctx 4096 --json"
echo ""
