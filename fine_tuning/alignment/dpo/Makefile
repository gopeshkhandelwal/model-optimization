# Makefile for DPO Training with Gaudi3
.PHONY: help build run clean logs run-pipeline run-pipeline-fast

# Default target
help:
	@echo "Available targets:"
	@echo "  build     - Build the Docker image for DPO training"
	@echo "  run       - Run the Docker container (requires HF_TOKEN)"
	@echo "  run-pipeline - Build+Run container then execute pipeline script (HF_TOKEN required)"
	@echo "  run-fast  - Run container with FAST_MODE=1 for ultra-fast demo"
	@echo "  run-pipeline-fast - Same as run-pipeline but FAST_MODE=1"
	@echo "  clean     - Clean up Docker images and containers"
	@echo "  logs      - Show container logs"
	@echo ""
	@echo "Usage examples:"
	@echo "  make build"
	@echo "  make run HF_TOKEN=your_token_here"
	@echo "  make run HF_TOKEN=your_token_here MODEL_NAME=google/gemma-3-270m"
	@echo "  make run-fast HF_TOKEN=your_token_here MODEL_NAME=google/gemma-3-270m"
	@echo ""
	@echo "FAST_MODE: Set FAST_MODE=1 or use run-fast target to enable --demo_fast flags in pipeline."

# Docker image name
IMAGE_NAME = dpo-gaudi3-training
CONTAINER_NAME = dpo-gaudi3-container

# Build Docker image
build:
	@echo "Building Docker image: $(IMAGE_NAME)"
	docker build -t $(IMAGE_NAME) .
	@echo "Docker image built successfully!"

# Run Docker container
run:
	@if [ -z "$(HF_TOKEN)" ]; then \
		echo "Error: HF_TOKEN is required. Usage: make run HF_TOKEN=your_token_here"; \
		exit 1; \
	fi
	@echo "Starting Docker container: $(CONTAINER_NAME)"
	@echo "HF_TOKEN: $(HF_TOKEN)"
	@if [ -n "$(MODEL_NAME)" ]; then \
		echo "MODEL_NAME: $(MODEL_NAME)"; \
	fi
	docker run -it --rm \
		--name $(CONTAINER_NAME) \
		--runtime=habana \
		-e HABANA_VISIBLE_DEVICES=all \
		-e OMPI_MCA_btl_vader_single_copy_mechanism=none \
		-e HF_TOKEN=$(HF_TOKEN) \
		$(if $(MODEL_NAME),-e MODEL_NAME=$(MODEL_NAME),) \
		--cap-add=sys_nice \
		--net=host \
		--ipc=host \
		-v /opt/habanalabs/:/opt/habanalabs/ \
		-v /var/log/habana_logs/:/var/log/habana_logs/ \
		$(IMAGE_NAME)

run-fast:
	@if [ -z "$(HF_TOKEN)" ]; then \
		echo "Error: HF_TOKEN is required. Usage: make run-fast HF_TOKEN=your_token_here MODEL_NAME=..."; \
		exit 1; \
	fi
	@echo "Starting FAST demo Docker container: $(CONTAINER_NAME)"
	docker run -it --rm \
		--name $(CONTAINER_NAME) \
		--runtime=habana \
		-e HABANA_VISIBLE_DEVICES=all \
		-e OMPI_MCA_btl_vader_single_copy_mechanism=none \
		-e HF_TOKEN=$(HF_TOKEN) \
		-e FAST_MODE=1 \
		$(if $(MODEL_NAME),-e MODEL_NAME=$(MODEL_NAME),) \
		--cap-add=sys_nice \
		--net=host \
		--ipc=host \
		-v /opt/habanalabs/:/opt/habanalabs/ \
		-v /var/log/habana_logs/:/var/log/habana_logs/ \
		$(IMAGE_NAME)

run-pipeline: build
	@if [ -z "$(HF_TOKEN)" ]; then \
		echo "Error: HF_TOKEN is required. Usage: make run-pipeline HF_TOKEN=your_token_here MODEL_NAME=google/gemma-3-270m"; \
		exit 1; \
	fi
	@if [ -z "$(MODEL_NAME)" ]; then \
		echo "MODEL_NAME not set. Defaulting to google/gemma-3-270m"; \
		MODEL_NAME=google/gemma-3-270m; \
	fi
	@echo "Running full DPO pipeline inside container (standard mode) MODEL_NAME=$(MODEL_NAME)"
	docker run --rm \
		--name $(CONTAINER_NAME) \
		--runtime=habana \
		-e HABANA_VISIBLE_DEVICES=all \
		-e OMPI_MCA_btl_vader_single_copy_mechanism=none \
		-e HF_TOKEN=$(HF_TOKEN) \
		-e MODEL_NAME=$(MODEL_NAME) \
		--cap-add=sys_nice \
		--net=host \
		--ipc=host \
		-v /opt/habanalabs/:/opt/habanalabs/ \
		-v /var/log/habana_logs/:/var/log/habana_logs/ \
		$(IMAGE_NAME) \
		bash -c "chmod +x dpo_pipeline_sanity.sh && ./dpo_pipeline_sanity.sh $(MODEL_NAME)"

run-pipeline-fast: build
	@if [ -z "$(HF_TOKEN)" ]; then \
		echo "Error: HF_TOKEN is required. Usage: make run-pipeline-fast HF_TOKEN=your_token_here MODEL_NAME=google/gemma-3-270m"; \
		exit 1; \
	fi
	@if [ -z "$(MODEL_NAME)" ]; then \
		echo "MODEL_NAME not set. Defaulting to google/gemma-3-270m"; \
		MODEL_NAME=google/gemma-3-270m; \
	fi
	@echo "Running FAST DPO pipeline inside container (FAST_MODE=1) MODEL_NAME=$(MODEL_NAME)"
	docker run --rm \
		--name $(CONTAINER_NAME) \
		--runtime=habana \
		-e HABANA_VISIBLE_DEVICES=all \
		-e OMPI_MCA_btl_vader_single_copy_mechanism=none \
		-e HF_TOKEN=$(HF_TOKEN) \
		-e FAST_MODE=1 \
		-e MODEL_NAME=$(MODEL_NAME) \
		--cap-add=sys_nice \
		--net=host \
		--ipc=host \
		-v /opt/habanalabs/:/opt/habanalabs/ \
		-v /var/log/habana_logs/:/var/log/habana_logs/ \
		$(IMAGE_NAME) \
		bash -c "chmod +x dpo_pipeline_sanity.sh && ./dpo_pipeline_sanity.sh $(MODEL_NAME) --fast"

# Clean up Docker resources
clean:
	@echo "Cleaning up Docker resources..."
	-docker stop $(CONTAINER_NAME) 2>/dev/null || true
	-docker rm $(CONTAINER_NAME) 2>/dev/null || true
	-docker rmi $(IMAGE_NAME) 2>/dev/null || true
	@echo "Cleanup completed!"

# Show container logs
logs:
	docker logs $(CONTAINER_NAME)

# Quick test run (builds and runs with minimal setup)
test: build
	@if [ -z "$(HF_TOKEN)" ]; then \
		echo "Error: HF_TOKEN is required. Usage: make test HF_TOKEN=your_token_here"; \
		exit 1; \
	fi
	@echo "Running quick test with DPO pipeline..."
	docker run --rm \
		--runtime=habana \
		-e HABANA_VISIBLE_DEVICES=all \
		-e OMPI_MCA_btl_vader_single_copy_mechanism=none \
		-e HF_TOKEN=$(HF_TOKEN) \
		-e MODEL_NAME=google/gemma-3-270m \
		--cap-add=sys_nice \
		--net=host \
		--ipc=host \
		-v /opt/habanalabs/:/opt/habanalabs/ \
		-v /var/log/habana_logs/:/var/log/habana_logs/ \
		$(IMAGE_NAME) \
		bash -c "chmod +x dpo_pipeline_sanity.sh && MODEL_NAME=google/gemma-3-270m ./dpo_pipeline_sanity.sh"