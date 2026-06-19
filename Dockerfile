FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Install python requirements
COPY requirements-colab.txt .
RUN pip install --no-cache-dir -r requirements-colab.txt

# Copy source code
COPY . .

# Install package
RUN pip install -e .

CMD ["/bin/bash"]
