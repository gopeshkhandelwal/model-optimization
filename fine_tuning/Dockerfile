FROM vault.habana.ai/gaudi-docker/1.21.0/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest

# Set working directory
WORKDIR /workspace

# Upgrade pip and install core Python dependencies
RUN pip install --upgrade pip && \
    pip install \
        optimum-habana==1.18.1 \
        transformers==4.51.0 \
        datasets==2.19.2 \
        pyarrow \
        rouge_score \
        nltk \
        evaluate \
        peft==0.12.0 \
        trl==0.9.6 \
        accelerate>=0.33 \
        sentencepiece!=0.1.92 \
        scipy \
        scikit-learn==1.5.2 \
        protobuf \
        torch>=1.3 \
        tyro

RUN pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.21.0

# Copy all POC step scripts into container
COPY *.py /workspace/

# Copy rlhf directory structure
COPY rlhf/ /workspace/rlhf/

# Default command
CMD ["/bin/bash"]

# Usage tip (multi-core + memory + Gaudi runtime)
# docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all \
#     -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host \
#     --ipc=host bigpatent-gemma3-habana
