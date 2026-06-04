# Detectron2 + custom transformers + HuggingFace data submodule = fragile install.
# This image pins a known-good combination: CUDA 11.8 + PyTorch 2.1 + Python 3.10,
# which has prebuilt Detectron2 wheels and is compatible with A100 (sm_80) and
# H100 (sm_90) GPUs.
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0" \
    FORCE_CUDA=1

# System packages:
#   git, git-lfs        -- HuggingFace dataset submodule (LFS-backed)
#   build-essential     -- compiling Detectron2 / segment-anything from source
#   libgl1, libglib2.0-0 -- runtime deps for opencv-python
#   ca-certificates     -- TLS for git+https clones
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        git-lfs \
        build-essential \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
    && git lfs install --system \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python deps. torch/torchvision are already in the base image; stripping
# them from the requirements list prevents pip from upgrading to a CUDA-mismatched
# wheel and breaking the Detectron2 build.
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip \
    && grep -vE '^[[:space:]]*(torch|torchvision)([[:space:]]|$)' /tmp/requirements.txt > /tmp/req-no-torch.txt \
    && pip install -r /tmp/req-no-torch.txt

# Project code and data are expected to be bind-mounted at /workspace at run time,
# e.g. `docker run --gpus all -v $PWD:/workspace tumor-seg:latest ...`
CMD ["/bin/bash"]
