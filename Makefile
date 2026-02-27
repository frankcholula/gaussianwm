IMAGE_NAME := gwm
IMAGE_TAG := latest
DOCKERHUB_USER := frankcholula
CONTAINER_NAME := gwm-container
GPUS ?= all
CUDA_ARCHS ?= 8.0;8.6

PWD := $(shell pwd)
DATA_DIR := $(PWD)/data
LOGS_DIR := $(PWD)/logs
CHECKPOINTS_DIR := $(PWD)/third_party/splatt3r/checkpoints

# Mount specific folders to avoid overwriting compiled CUDA extensions in the container
DOCKER_RUN_FLAGS := --rm --gpus $(GPUS) \
	--name $(CONTAINER_NAME) \
	--ipc=host \
	--shm-size=16g \
	-v $(PWD)/gaussianwm:/app/gaussianwm \
	-v $(PWD)/configs:/app/configs \
	-v $(PWD)/scripts:/app/scripts \
	-v $(PWD)/tests:/app/tests \
	-v $(DATA_DIR):/app/data \
	-v $(LOGS_DIR):/app/logs \
	-v $(CHECKPOINTS_DIR):/app/third_party/splatt3r/checkpoints \
	-e CUDA_VISIBLE_DEVICES \
	-e WANDB_API_KEY \
	-e GWM_PATH=/app

.PHONY: build run shell test clean push

build:
	docker build --build-arg CUDA_ARCHS="$(CUDA_ARCHS)" -t $(IMAGE_NAME):$(IMAGE_TAG) .

run:
	@mkdir -p $(DATA_DIR) $(LOGS_DIR) $(CHECKPOINTS_DIR)
	docker run -it $(DOCKER_RUN_FLAGS) $(IMAGE_NAME):$(IMAGE_TAG) /bin/bash

test:
	docker run $(DOCKER_RUN_FLAGS) $(IMAGE_NAME):$(IMAGE_TAG) python tests/test_install.py

push:
	docker tag $(IMAGE_NAME):$(IMAGE_TAG) $(DOCKERHUB_USER)/$(IMAGE_NAME):$(IMAGE_TAG)
	docker push $(DOCKERHUB_USER)/$(IMAGE_NAME):$(IMAGE_TAG)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ 2>/dev/null || true
