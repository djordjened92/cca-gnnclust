FROM nvcr.io/nvidia/cuda:11.7.0-runtime-ubuntu20.04
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && echo PATH="/root/miniconda3/bin":$PATH >> .bashrc \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && conda init bash \
    && exec bash

RUN conda install python=3.10
RUN conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
RUN conda install -c pytorch -c nvidia faiss-gpu=1.7.4 mkl=2021 blas=1.0=mkl
RUN conda install -c dglteam/label/cu117 dgl