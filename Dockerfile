FROM python:3.8-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libpoco-dev \
    libeigen3-dev \
    libfmt-dev \
    wget \
    ca-certificates \
    sudo \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

# Clone libfranka repo and build using included script
ARG LIBFRANKA_VERSION=0.8.0
# RUN git clone --recursive https://github.com/frankaemika/libfranka.git && \
#     cd libfranka && \
#     if [ -z "$LIBFRANKA_VERSION" ]; then ./scripts/build_libfranka.sh; else ./scripts/build_libfranka.sh $LIBFRANKA_VERSION; fi && \
#     cd .. && rm -rf libfranka

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

# Accept Terms of Service for main channels
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Update conda and install packages
RUN conda update -n base -c defaults conda -y

# # Install PyTorch (default CPU-only, add ARG to install CUDA version)
# ARG INSTALL_CUDA=false
# RUN if [ "$INSTALL_CUDA" = "true" ]; then \
#         pip3 install pytorch torchvision torchaudio pytorch-cuda=11.8 ; \
#     else \
#         pip3 install pytorch torchvision torchaudio cpuonly; \
#     fi

RUN git config --global url."https://github.com/".insteadOf "git@github.com:"
RUN apt-get update && apt-get install -y \
    libspdlog-dev \
    # also needed: add build-essential cmake ninja-build clang if not installed
    build-essential cmake ninja-build clang


# Clone polymetis repo and build from source with cmake options
RUN git clone --recursive https://github.com/facebookresearch/fairo.git && \
    cd fairo/polymetis && conda env create -f ./polymetis/environment.yml && \
    conda run -n polymetis-local pip install -e ./polymetis

# RUN sed -i '1i#include <string>' /opt/fairo/polymetis/polymetis/src/clients/franka_panda_client/third_party/libfranka/include/franka/control_tools.h
RUN apt-get update && apt-get install -y \
    libeigen3-dev \
    libboost-all-dev \
    cmake \
    python3-pybind11 \
    python3-numpy \
    python3-dev 
    # other dependencies

# Build Pinocchio from source
# RUN conda run -n polymetis-local conda install pinocchio -c conda-forge
RUN sed -i '1i#include <string>' /opt/fairo/polymetis/polymetis/src/clients/franka_panda_client/third_party/libfranka/include/franka/control_tools.h
RUN sed -i '1i#include <cstddef>' /opt/fairo/polymetis/polymetis/torch_isolation/include/torch_server_ops.hpp

RUN cd fairo/polymetis && conda run -n polymetis-local ./scripts/build_libfranka.sh ${LIBFRANKA_VERSION} && \
    mkdir -p polymetis/build && cd polymetis/build && \
    conda run -n polymetis-local cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_FRANKA=ON -DBUILD_TESTS=OFF -DBUILD_DOCS=OFF && \
    conda run -n polymetis-local make -j$(nproc)

# Install dependencies for librealsense build and runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libusb-1.0-0-dev \
    pkg-config \
    libglfw3-dev \
    libssl-dev \
    usbutils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

# Clone librealsense and build
RUN git clone https://github.com/IntelRealSense/librealsense.git

# RUN cd librealsense && \
#     ./scripts/setup_udev_rules.sh
RUN mkdir -p /etc/udev/rules.d 
# RUN cd librealsense && ./scripts/setup_udev_rules.sh
RUN cd librealsense && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && \
    make install && ldconfig

# Optional: If you want to build Python bindings for librealsense (pyrealsense2)
RUN cd /opt/librealsense/build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_BINDINGS=ON && \
    make -j$(nproc) && \
    make install && ldconfig

# Note: You may need to pip install pyrealsense2 bindings separately inside your conda env or system python
# RUN conda run -n polymetis-local pip install pyrealsense2

WORKDIR /app

CMD ["/bin/bash"]
