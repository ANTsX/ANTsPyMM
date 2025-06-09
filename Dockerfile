FROM tensorflow/tensorflow:2.16.2


# Set environment variables for optimal threading
ENV TF_NUM_INTEROP_THREADS=8 \
    TF_NUM_INTRAOP_THREADS=8 \
    ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=8 \
    OPENBLAS_NUM_THREADS=8 \
    MKL_NUM_THREADS=8

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python libraries
RUN pip install --upgrade pip \
    && pip install \
        numpy \
        pandas \
        scipy \
        matplotlib \
        scikit-learn \
        ipython \
        jupyterlab
#        # Optional:
#        antspynet \
#        antspymm \
#        siq
#
# Clone the ANTsPyMM repository
# RUN git clone https://github.com/ANTsX/ANTsPyMM.git /workspace/ANTsPyMM
#
# Optional: Run reference test script
# RUN python /workspace/ANTsPyMM/tests/test_reference_run.py

# Default command
CMD ["bash"]
