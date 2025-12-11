#!/bin/bash

# Get the absolute path of the project root
PROJECT_ROOT="$(pwd)"

echo "Starting Mamba Container on GPU 0 (NVIDIA A2)..."

docker run --gpus '"device=0"' \
    --name mamba_dev_container \
    --pull=always \
    --rm -it \
    --shm-size=16g \
    -v "${PROJECT_ROOT}:/app" \
    aimilefth/mamba-a2