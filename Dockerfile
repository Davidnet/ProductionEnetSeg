FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04  
WORKDIR /usr/src/app
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        libatlas-base-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        python-dev \
        python-numpy \
        python-pip \
        python-setuptools \
        python-scipy && \
    rm -rf /var/lib/apt/lists/*
RUN git clone --recursive https://github.com/TimoSaemann/ENet.git
RUN cd ENet/caffe-enet && mkdir build && cd build && cmake .. && make all -j2 && make pycaffe
RUN wget https://modeldepot.io/assets/uploads/models/models/b646b316-157d-4e06-95d0-0af0db2481a4_cityscapes_weights.caffemodel -O cityscapes_weights.caffemodel
RUN apt-get update && apt-get install -y --no-install-recommends \
    python-matplotlib \
    python-pil && \
    rm -rf /var/lib/apt/lists/*

RUN pip install -U scikit-image protobuf opencv-python