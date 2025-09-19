FROM vault.habana.ai/gaudi-docker/1.21.0/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest

# Set working directory
WORKDIR /workspace

# Upgrade pip and install core Python dependencies
RUN pip install --upgrade pip && \
    pip install \
        optimum-habana==1.18.1 \
        transformers==4.51.0 \
        datasets \
        pyarrow \
        rouge_score \
        nltk \
        evaluate \
        peft \
        trl==0.19.1 \
        accelerate>=0.33

# Copy all POC step scripts into container
COPY *.py /workspace/

# Default command
CMD ["/bin/bash"]

# Usage tip (multi-core + memory + Gaudi runtime)
# docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all \
#     -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host \
#     --ipc=host bigpatent-gemma3-habana
