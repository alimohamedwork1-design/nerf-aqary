FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# System deps + Python + COLMAP
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    git ca-certificates zip ffmpeg \
    build-essential cmake ninja-build \
    colmap \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip

# Pin numpy <2 (critical)
RUN pip install --no-cache-dir "numpy<2"

# PyTorch CUDA 11.8
RUN pip install --no-cache-dir \
    torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu118

# CUDA env
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV TORCH_CUDA_ARCH_LIST="8.6;8.9"

# Sanity checks
RUN which nvcc && nvcc --version
RUN which colmap && colmap --version || true
RUN python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# Clone gaussian-splatting
RUN git clone --depth 1 https://github.com/graphdeco-inria/gaussian-splatting.git /workspace/gaussian-splatting
WORKDIR /workspace/gaussian-splatting
RUN git submodule update --init --recursive

# Python deps
RUN pip install --no-cache-dir \
    tqdm pillow imageio scipy matplotlib opencv-python-headless plyfile

# Re-assert numpy pin
RUN pip install --no-cache-dir --force-reinstall "numpy<2"

# Build CUDA extensions
RUN pip install --no-cache-dir "setuptools<70" wheel pybind11
RUN pip install --no-cache-dir --no-build-isolation ./submodules/diff-gaussian-rasterization
RUN pip install --no-cache-dir --no-build-isolation ./submodules/simple-knn

# Serverless deps
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Sanity checks
RUN python3 -c "import numpy as np; print('NUMPY', np.__version__)" \
    && python3 -c "import diff_gaussian_rasterization; print('DGR_OK')" \
    && python3 -c "import simple_knn; print('SKNN_OK')" \
    && python3 -c "from plyfile import PlyData; print('PLY_OK')"

COPY handler.py /app/handler.py

CMD ["python3", "/app/handler.py"]
