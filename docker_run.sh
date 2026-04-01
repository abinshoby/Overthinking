#!/bin/bash
DEBUG_PORT=5697
JUPYTER_PORT=8788
IMAGE_NAME=detect

# Run the docker container in detached mode and capture its container ID
CONTAINER_ID=$(docker run --ipc=host --shm-size=10g --gpus all --runtime=nvidia \
  -p "${DEBUG_PORT}:$DEBUG_PORT" \
  -p "${JUPYTER_PORT}:$JUPYTER_PORT" \
  --rm \
  -v /mnt/16T/abin/data:/workspace/data \
  -v /mnt/16T/abin/Overthinking-Causes-Hallucination/:/workspace/code \
  -dit "$IMAGE_NAME" /bin/bash)

echo "Container started: $CONTAINER_ID"

# start Jupyter inside container in background
echo "Starting Jupyter Notebook on port ${JUPYTER_PORT}..."
docker exec -d "$CONTAINER_ID" bash -lc "nohup jupyter notebook  --port=${JUPYTER_PORT} > /workspace/code/jupyter_${IMAGE_NAME}.log 2>&1 &"

# Exec into the container (interactive bash)
docker exec -it "$CONTAINER_ID" /bin/bash

