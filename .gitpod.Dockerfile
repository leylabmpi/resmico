FROM condaforge/mambaforge
RUN mkdir /sys/fs/cgroup/systemd && \
    mount -t cgroup -o none,name=systemd cgroup /sys/fs/cgroup/systemd && \
    apt-get update && \
    apt-get install -y gcc-9 g++-9