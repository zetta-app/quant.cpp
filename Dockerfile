FROM ubuntu:22.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        g++ \
        make \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy project source (see .dockerignore for exclusions)
COPY . /turboquant
WORKDIR /turboquant

# Build the library, tools, and tests
RUN cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DTQ_BUILD_TESTS=ON \
        -DTQ_BUILD_BENCH=ON \
    && cmake --build build -j$(nproc)

# Run the test suite
RUN ctest --test-dir build --output-on-failure

# Default entrypoint: the tq_run inference CLI
# Usage: docker run turboquant models/model.tqm -p "Hello" -k turbo_kv_1b
ENTRYPOINT ["./build/tq_run"]
