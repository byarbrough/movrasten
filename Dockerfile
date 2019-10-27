FROM tensorflow/tensorflow:1.15.0-py3
# this is based on https://github.com/IntelAI/OpenVINO-model-server/blob/master/Dockerfile

WORKDIR /app

# basic install
RUN apt-get update && apt-get install -y --no-install-recommends \
            ca-certificates \
            lsb-core \
            sudo

# download OpenVINO
RUN curl -o GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
RUN apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
RUN echo "deb https://apt.repos.intel.com/openvino/2019/ all main" > /etc/apt/sources.list.d/intel-openvino-2019.list
# install OpenVINO
RUN apt-get update && apt-get install -y intel-openvino-dev-ubuntu18-2019.2.242
RUN cd /opt/intel/openvino/deployment_tools/model_optimizer/install_prerequisites/ && ./install_prerequisites_tf.sh
RUN /opt/intel/openvino/bin/setupvars.sh

# additional python dependencies
copy requirements.txt /app
RUN pip --no-cache-dir install -r requirements.txt