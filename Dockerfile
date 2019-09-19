FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu16.04

RUN apt-get clean && apt-get update && apt-get install -y locales
RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen
ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8
ENV SHELL /bin/bash

RUN apt-get update && \
    apt-get install -y curl bzip2 xvfb ffmpeg git libxrender1

WORKDIR /aaio

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda clean -ya && \
     /opt/conda/bin/conda create -n python36 python=3.6 numpy

ENV PATH /opt/conda/envs/python36/bin:/opt/conda/envs/bin:$PATH

RUN pip install tensorflow-gpu==1.14
RUN pip install animalai

ENV HTTP_PROXY ""
ENV HTTPS_PROXY ""
ENV http_proxy ""
ENV https_proxy ""

################################################################################################
# YOUR COMMANDS GO HERE


