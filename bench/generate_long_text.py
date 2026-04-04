#!/usr/bin/env python3
"""
generate_long_text.py — Prepare long-context perplexity evaluation corpus

Downloads WikiText-2 test set (or uses a local copy) and splits it into
token-length-calibrated files for benchmarking PPL at 1K, 4K, 8K, 16K tokens.

Token estimation: ~1.3 tokens per whitespace-delimited word (empirical average
for LLaMA/GPT-style BPE tokenizers on English prose). We generate files with
a safety margin so the tokenizer produces at least the target token count.

Usage:
    python3 bench/generate_long_text.py [--source <path>] [--output-dir <dir>]

If no --source is given, tries (in order):
    1. bench/data/wikitext2_test.txt  (local copy)
    2. Download from Hugging Face datasets

Output:
    bench/data/ppl_1k.txt    (~1,024 tokens)
    bench/data/ppl_4k.txt    (~4,096 tokens)
    bench/data/ppl_8k.txt    (~8,192 tokens)
    bench/data/ppl_16k.txt   (~16,384 tokens)
"""

import argparse
import os
import sys
import textwrap

# Approximate tokens-per-word ratio for BPE tokenizers on English text.
# LLaMA tokenizer averages ~1.3 tokens/word on WikiText-2.
TOKENS_PER_WORD = 1.3

# Target token counts and the output filenames
TARGETS = [
    (1024,  "ppl_1k.txt"),
    (4096,  "ppl_4k.txt"),
    (8192,  "ppl_8k.txt"),
    (16384, "ppl_16k.txt"),
]

WIKITEXT2_URL = (
    "https://huggingface.co/datasets/wikitext/resolve/refs%2Fconvert%2Fparquet/"
    "wikitext-2-raw-v1/test/0000.parquet"
)


def load_wikitext2_local(path):
    """Load a local plain-text WikiText-2 file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def download_wikitext2(cache_path):
    """
    Download WikiText-2 test set.  Tries two strategies:
      1. HuggingFace datasets library (if installed)
      2. Raw HTTP download of the plain-text version from HuggingFace Hub
    Returns the text as a string and saves to cache_path.
    """
    # Strategy 1: datasets library
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n".join(row["text"] for row in ds if row["text"].strip())
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(text)
        return text
    except ImportError:
        pass

    # Strategy 2: urllib (standard library only)
    import urllib.request
    import json

    # Try the raw text files hosted on HuggingFace
    raw_url = (
        "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/"
        "data/wikitext-2/test.txt"
    )
    alt_urls = [
        raw_url,
        "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip",
    ]

    for url in alt_urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "quant.cpp-bench/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
                if url.endswith(".zip"):
                    import zipfile
                    import io
                    with zipfile.ZipFile(io.BytesIO(data)) as zf:
                        for name in zf.namelist():
                            if "test" in name and name.endswith(".txt"):
                                text = zf.read(name).decode("utf-8")
                                break
                        else:
                            # Take the largest text file
                            name = max(zf.namelist(), key=lambda n: zf.getinfo(n).file_size)
                            text = zf.read(name).decode("utf-8")
                else:
                    text = data.decode("utf-8")

                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(text)
                return text
        except Exception:
            continue

    return None


def clean_wikitext(text):
    """
    Remove WikiText markup artifacts:
      - Lines starting with ' = ' (section headers)
      - Empty lines (collapse to single newline)
      - Leading/trailing whitespace per line

    Keeps paragraph structure intact for natural PPL evaluation.
    """
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Skip empty lines and section headers like " = Title = "
        if not stripped:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue
        if stripped.startswith("= ") and stripped.endswith(" ="):
            continue
        cleaned.append(stripped)

    return "\n".join(cleaned).strip()


def estimate_words_for_tokens(target_tokens):
    """
    Estimate number of whitespace words needed to produce target_tokens
    after BPE tokenization. Adds a 10% safety margin.
    """
    return int(target_tokens / TOKENS_PER_WORD * 1.10)


def split_to_word_count(text, n_words):
    """Extract the first n_words whitespace-delimited words from text."""
    words = text.split()
    if len(words) < n_words:
        return None, len(words)
    # Find a sentence boundary near n_words for cleaner splits
    # Look backwards from n_words for a period
    end = n_words
    for i in range(n_words, max(n_words - 50, 0), -1):
        if words[i - 1].endswith((".","!","?")):
            end = i
            break
    return " ".join(words[:end]), end


def main():
    parser = argparse.ArgumentParser(
        description="Generate long-context PPL evaluation corpus from WikiText-2"
    )
    parser.add_argument(
        "--source", "-s",
        help="Path to existing WikiText-2 plain text file"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Output directory (default: bench/data/)"
    )
    parser.add_argument(
        "--tokens-per-word", type=float, default=TOKENS_PER_WORD,
        help=f"Tokens-per-word ratio for estimation (default: {TOKENS_PER_WORD})"
    )
    args = parser.parse_args()

    # Determine output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = args.output_dir or os.path.join(script_dir, "data")
    os.makedirs(output_dir, exist_ok=True)

    # Load source text
    text = None
    source_path = args.source

    if source_path and os.path.isfile(source_path):
        print(f"Loading source: {source_path}")
        text = load_wikitext2_local(source_path)
    else:
        # Try local wikitext2_test.txt first
        local_wt2 = os.path.join(script_dir, "data", "wikitext2_test.txt")
        if os.path.isfile(local_wt2):
            print(f"Using local WikiText-2: {local_wt2}")
            text = load_wikitext2_local(local_wt2)
        else:
            # Download
            cache_path = os.path.join(output_dir, "wikitext2_test.txt")
            print("Downloading WikiText-2 test set...")
            text = download_wikitext2(cache_path)
            if text is None:
                print(
                    "ERROR: Could not download WikiText-2.\n"
                    "Please provide a text file manually:\n"
                    f"  python3 {sys.argv[0]} --source <your_text_file.txt>",
                    file=sys.stderr
                )
                sys.exit(1)
            print(f"Cached to: {cache_path}")

    # Clean the text
    text = clean_wikitext(text)
    total_words = len(text.split())
    print(f"Source: {total_words:,} words (~{int(total_words * args.tokens_per_word):,} estimated tokens)")
    print()

    # Generate splits
    print(f"{'Target':>8s}  {'Est Words':>10s}  {'Actual Words':>13s}  {'File':>20s}")
    print(f"{'------':>8s}  {'---------':>10s}  {'------------':>13s}  {'----':>20s}")

    for target_tokens, filename in TARGETS:
        needed_words = estimate_words_for_tokens(target_tokens)
        chunk, actual_words = split_to_word_count(text, needed_words)

        if chunk is None:
            est_tokens = int(actual_words * args.tokens_per_word)
            print(
                f"{target_tokens:>8,d}  {needed_words:>10,d}  "
                f"{'SKIP':>13s}  "
                f"(source too short: {actual_words:,} words = ~{est_tokens:,} tokens)"
            )
            continue

        out_path = os.path.join(output_dir, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(chunk)
            f.write("\n")

        est_tokens = int(actual_words * args.tokens_per_word)
        print(f"{target_tokens:>8,d}  {needed_words:>10,d}  {actual_words:>13,d}  {filename:>20s}  (~{est_tokens:,} tokens)")

    print()
    print(f"Output directory: {output_dir}")
    print()
    print("Verify actual token counts with your model's tokenizer:")
    print(f"  for f in {output_dir}/ppl_*.txt; do")
    print(f"    echo \"$f: $(wc -w < \"$f\") words\"")
    print(f"  done")


if __name__ == "__main__":
    main()
