FROM ubuntu:22.04

WORKDIR /work
EXPOSE 22

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -yq tzdata && \
    ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata


RUN apt-get update && apt-get install -y openssh-server git && \
    mkdir /var/run/sshd && \
    echo 'root:root' | chpasswd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

RUN apt install -y curl wget apt install lsb-release software-properties-common gnupg \
                    ninja-build


CMD ["/usr/sbin/sshd", "-D"]
