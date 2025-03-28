# ✅ Base image with CUDA 12.4
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Set environment vars
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH="/opt/conda/bin:$PATH" \
    PYTHONUNBUFFERED=1

# 🔧 Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl build-essential ca-certificates \
    libglib2.0-0 libsm6 libxext6 libxrender-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 🐍 Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && rm miniconda.sh

# 🧪 Create Conda env and install CUDA toolkit & sparsehash
RUN conda create -n spatiallm python=3.11 -y && \
    conda run -n spatiallm conda install -c nvidia/label/cuda-12.4.0 cuda-toolkit -y && \
    conda run -n spatiallm conda install -c conda-forge sparsehash -y && \
    conda clean -afy

# 📦 Install Poetry
RUN conda run -n spatiallm pip install poetry && \
    conda run -n spatiallm poetry config virtualenvs.create false --local

# 📁 Clone SpatialLM
WORKDIR /workspace
RUN git clone https://github.com/manycore-research/SpatialLM.git
WORKDIR /workspace/SpatialLM

# 📜 Install dependencies via Poetry
RUN conda run -n spatiallm poetry install

# 🛠️ Build torchsparse (slow, but necessary)
RUN conda run -n spatiallm poetry run poe install-torchsparse

# 📂 Create folders for model/testset/output if needed
RUN mkdir -p /workspace/models /workspace/output /workspace/data

# 🧼 Set entrypoint
CMD ["/bin/bash"]
