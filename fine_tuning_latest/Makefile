
# Makefile for Fine-Tuning Workflow

.PHONY: build run

DOCKER_IMAGE=gsm8k-llama-3-8b-habana
ENV_FLAGS=\
	-e HF_TOKEN=$(HF_TOKEN)

DOCKER_RUN=docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host $(ENV_FLAGS) $(DOCKER_IMAGE)

build:
	docker build -t $(DOCKER_IMAGE) .

run:
ifeq ($(HF_TOKEN),)
	@echo "ERROR: HF_TOKEN is not provided. Please provide your Hugging Face token: make run HF_TOKEN=your_hf_token_here" && exit 1
else
	$(DOCKER_RUN)
endif