#!/bin/bash

# Build the docker image with the tag 'mamba-a2'
docker buildx build -f Dockerfile --tag aimilefth/mamba-a2 --push  .