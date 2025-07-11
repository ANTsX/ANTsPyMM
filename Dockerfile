FROM tensorflow/tensorflow:2.17.0

ENV HOME=/workspace
WORKDIR $HOME

# Set environment variables for optimal threading
ENV TF_NUM_INTEROP_THREADS=8 \
    TF_NUM_INTRAOP_THREADS=8 \
    ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=8 \
    OPENBLAS_NUM_THREADS=8 \
    MKL_NUM_THREADS=8

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
    psrecord \
    numpy \
    pandas \
    scipy \
    matplotlib \
    scikit-learn \
    ipython \
    jupyterlab \
    antspyx==0.6.1 \
    antspynet==0.3.1 \
    antspyt1w==1.1.3 \
    antspymm==1.6.4 \
    siq==0.4.1

# for downloading example data from open neuro
RUN pip3 --no-cache-dir install --upgrade awscli
###########
#
RUN git clone https://github.com/stnava/ANTPD_antspymm.git ${HOME}/ANTPD_antspymm
RUN python ${HOME}/ANTPD_antspymm/src/get_antsxnet_data.py ${HOME}/.keras
# RUN cd ${HOME}/ANTPD_antspymm && bash src/download_docker.sh
# data is in ${HOME}/.keras/, ~/.antspymm and bids folders
# Default command
CMD ["bash"]

