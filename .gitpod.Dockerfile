FROM: condaforge/mambaforge
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get install -y gcc-9 && \
    apt-get install -y cmake-9