# Docker Usage Guide

quant.cpp ships as a minimal Docker image (~10MB) built on Alpine Linux.
The binary is statically linked with zero runtime dependencies.

## Quick Start

### Build the image

```bash
docker build -t quant.cpp .
```

### Run inference

Mount a directory containing your GGUF model file and pass CLI arguments:

```bash
docker run -v ./models:/models quant.cpp /models/model.gguf -p "hello" -k uniform_4b -v q4
```

### Full example with all options

```bash
docker run -v ./models:/models quant.cpp \
    /models/model.gguf \
    -p "Once upon a time" \
    -n 512 \
    -k turbo_3b \
    -v q4 \
    -j 4 \
    -T 0.8
```

### Print model info

```bash
docker run -v ./models:/models quant.cpp /models/model.gguf --info
```

### Compute perplexity

```bash
docker run -v ./models:/models -v ./data:/data quant.cpp \
    /models/model.gguf --ppl /data/wikitext.txt -k polar_3b -v q4
```

## Docker Compose

The included `docker-compose.yml` provides a preconfigured inference service:

```bash
# Place your model at ./models/model.gguf, then:
docker compose up

# Override the prompt:
docker compose run inference /models/model.gguf -p "Your prompt here" -k turbo_3b -v q4
```

Edit `docker-compose.yml` to change the default model path, KV compression type,
or thread count.

## KV Compression Options

| Flag | Values | Description |
|------|--------|-------------|
| `-k` | `fp32`, `uniform_4b`, `uniform_2b`, `polar_3b`, `polar_4b`, `turbo_3b`, `turbo_4b` | Key cache quantization |
| `-v` | `fp16`, `q4`, `q2` | Value cache quantization |
| `-j` | integer | Thread count for matmul |

## Volume Mounts

Models are not baked into the image. Mount them at runtime:

- `/models` -- default mount point for GGUF model files
- Mount additional directories as needed (e.g., `/data` for perplexity evaluation)

## Image Size

The final image is approximately 10MB:
- Alpine base: ~7MB
- quant binary: ~500KB (statically linked, zero dependencies)
