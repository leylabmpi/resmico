FROM ubuntu:xenial
FROM condaforge/mambaforge
RUN apt-get update && \
    apt-get install -y gcc g++