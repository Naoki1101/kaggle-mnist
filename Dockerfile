FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    unzip \
    wget \
    vim \
    tmux

SHELL ["/bin/bash", "-c"]
ENV PATH /opt/conda/bin:$PATH

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.7.10-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN conda install pytorch torchvision cudatoolkit=10.1 -c pytorch && \
    conda install jupyter ipykernel matplotlib pandas pillow scipy scikit-learn \
          tqdm opencv joblib && \
    conda clean -p &&\
    conda clean -t &&\
    conda clean --yes --all
RUN conda clean --all

COPY requirements.txt .
RUN pip install -U pip && \
    pip install -r requirements.txt
