###################################################################################################
# Choose a source
###################################################################################################
ARG CUDA_VERSION=11.4.1
ARG UBUNTU_VERSION=20.04
ARG CUDA_FLAVOR=cudnn8-devel
FROM nvidia/cuda:${CUDA_VERSION}-${CUDA_FLAVOR}-ubuntu${UBUNTU_VERSION}

###################################################################################################
# Setup Args and Envs
###################################################################################################
ARG ARG_PROJECT_NAME
ENV PROJECT_NAME ${ARG_PROJECT_NAME}

ARG ARG_IMAGE_NAME
ENV CONTAINER_IMAGE_NAME ${ARG_IMAGE_NAME}
ENV IMAGE_NAME ${ARG_IMAGE_NAME}

ENV SOURCE_FOLDER 'docker/'
###################################################################################################
# Setup Base Environment
###################################################################################################
RUN DEBIAN_FRONTEND="noninteractive" apt-get update && apt-get install --no-install-recommends -y \
    python3-dev python3-pip python-is-python3 \
    systemctl \
    apt-utils \
    jq \
    git \ 
    wget curl \
    zip unzip \
    apt-transport-https \
    ca-certificates \
    libsm6 libxext6 libxrender-dev \
    freetds-dev freetds-bin unixodbc unixodbc-dev tdsodbc \
    gnupg \
    nano \
    pandoc \
    && apt-get autoremove -y

###################################################################################################
# Setup component tools
###################################################################################################
# pip =============================================================================================
COPY ${SOURCE_FOLDER}/requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Jupyter-lab =====================================================================================
ARG ARG_JUPYTER_PORT
ENV JUPYTER_PORT ${ARG_JUPYTER_PORT}
EXPOSE ${JUPYTER_PORT}

# Enable jupyter-lab to be run as a service
COPY ${SOURCE_FOLDER}/jupyter-lab.service /lib/systemd/system/jupyter-lab.service
RUN ex_path=`which python` && sed -i "s|REPLACE_TEXT|${ex_path}|" /lib/systemd/system/jupyter-lab.service 
RUN jupyter nbextension enable --py widgetsnbextension

# VScode (code-server) ============================================================================
ARG ARG_VSCODE_PORT
ENV VSCODE_PORT=${ARG_VSCODE_PORT}
EXPOSE ${VSCODE_PORT}

# Install code-server
RUN VSCODE_SERVER_VERSION=3.9.2 \
    && curl -fsSL https://code-server.dev/install.sh | sh -s -- --version ${VSCODE_SERVER_VERSION}

# Move the settings to the right spot   
RUN mkdir -p /root/.local/share/code-server/User
COPY ${SOURCE_FOLDER}/settings.json /root/.local/share/code-server/User/settings.json
RUN ex_path=`which python` && sed -i "s|REPLACE_PYTHON_PATH|${ex_path}|" /root/.local/share/code-server/User/settings.json
RUN ex_path=`which bash` && sed -i "s|REPLACE_BASH_PATH|${ex_path}|" /root/.local/share/code-server/User/settings.json

# Install the extensions
# Note at the time of build, 404 error for ms-python.python; thus, open-vsx
RUN code-server --verbose \
--install-extension ms-toolsai.jupyter \
# --install-extension ms-python.python@2022.0.1814523869 \
--install-extension mhutchie.git-graph
RUN SERVICE_URL=https://open-vsx.org/vscode/gallery ITEM_URL=https://open-vsx.org/vscode/item code-server --install-extension ms-python.python
RUN SERVICE_URL=https://open-vsx.org/vscode/gallery ITEM_URL=https://open-vsx.org/vscode/item code-server --install-extension redhat.vscode-didact

# Enable code-server to be run as a service
COPY ${SOURCE_FOLDER}/code-server.service /lib/systemd/system/code-server.service
RUN ex_path=`which code-server` && sed -i "s|REPLACE_TEXT|${ex_path}|" /lib/systemd/system/code-server.service

###################################################################################################
# System Setup - Create, Add, Copy, Run, and Clean
###################################################################################################
# Make some file folders
RUN mkdir -p /workspace/code /workspace/data

# Copy over code
COPY /src /workspace/code/src
COPY setup.py /workspace/code
COPY /examples /workspace/code/examples

# pip install the module
RUN pip install /workspace/code

# Add a README
COPY ${SOURCE_FOLDER}/docker_readme.didact.md /workspace/readme.didact.md

# Push Datetime build info
RUN date '+%Y.%m.%d %H:%M' > /tmp/build_dt.info

###################################################################################################
# Entrypoint
###################################################################################################
WORKDIR /workspace
COPY ${SOURCE_FOLDER}/entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["entrypoint.sh"]: