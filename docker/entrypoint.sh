#!/bin/bash
#################################################################################
# Results Management AI Environment
#################################################################################

# Define variables / defaults ####################################################
# Define colors (https://stackoverflow.com/questions/5947742/how-to-change-the-output-color-of-echo-in-linux)
NC='\033[0m'
RED='\033[0;31m'
GRN='\033[1;32m'
YLW='\033[1;33m'
BLU='\033[1;34m'
PRP='\033[1;35m'
CYN='\033[0;36m'
WHT='\033[1;37m'

# git defaults
git config --global credential.helper 'cache --timeout 28800'

# Determine and Display Container info ##########################################
sleep 1 # sometimes too fast and info doesn't pull correcly
export CONTAINER_ID=`cat /proc/self/cgroup | grep 'docker' | sed 's/^.*\///' | tail -n1 | cut -c1-12`

# Check whether we should go to IP address (remote) or localhost (ssh tunnel/local)
# This assumes $ip_addr is pushed to containerdock via -e ip_addr argument at run
if [[ -v ip_addr ]]
then
    LOCAL_HOSTNAME=${ip_addr}
else
    LOCAL_HOSTNAME='localhost'
fi

# Check which port(s) we should print - variables passed in .env file
if [[ -v VSCODE_PORT_CLIENT ]]
then
    VSCODE_PORT_PRINT=${VSCODE_PORT_CLIENT}
else
    VSCODE_PORT_PRINT=${VSCODE_PORT}
fi

if [[ -v JUPYTER_PORT_CLIENT ]]
then
    JUPYTER_PORT_PRINT=${JUPYTER_PORT_CLIENT}
else
    JUPYTER_PORT_PRINT=${JUPYTER_PORT}
fi

echo -e "${PRP}====================================================================================================${NC}"
echo -e "${PRP} Starting ${GRN}${IMAGE_NAME}${PRP} container (Built:${GRN} $(cat /tmp/build_dt.info)${PRP})${NC}"
echo -ne "${PRP} Container ID: ${WHT}${CONTAINER_ID}${NC}"
echo -e "\n${PRP} Container for: ${GRN}${PROJECT_NAME}${NC}"

# Start code server and jupyter-lab #############################################
echo -ne " ${PRP}Starting VS Code... ${NC}"
b_started=0
while (( ! b_started ))
do
    systemctl start code-server 
    count=`ps aux | grep -c 0.0.0.0:${VSCODE_PORT}`
    count=$(( ${count}-1 ))
    b_started=count
done
echo -e "${GRN}complete${NC}"

echo -ne " ${PRP}Starting Jupyter Lab... ${NC}"
b_started=0
while (( ! b_started ))
do
    systemctl start jupyter-lab
    count=`ps aux | grep -c port=${JUPYTER_PORT}`
    count=$(( ${count}-1 ))
    b_started=count
done
echo -e "${GRN}complete${NC}"

# Edit bashrc ###################################################################
echo "" >> ~/.bashrc
echo "" >> ~/.bashrc
echo "" >> ~/.bashrc

if [ "$HOSTNAME" == "$CONTAINER_ID" ]; then
    echo "export PS1=\"\[\033[38;5;10m\]\u@\[$(tput sgr0)\]\[\033[38;5;12m\][\[$(tput sgr0)\]\[\033[38;5;10m\]\h\[$(tput sgr0)\]\[\033[38;5;12m\]]\[$(tput sgr0)\]:\[$(tput sgr0)\]\[\033[38;5;12m\]\W\[$(tput sgr0)\]\\$ \[$(tput sgr0)\]\"" >> ~/.bashrc
else
    echo "export PS1=\"\[\033[38;5;10m\]\u@\h\[$(tput sgr0)\]\[\033[38;5;12m\][\[$(tput sgr0)\]\[\033[38;5;10m\]${CONTAINER_ID}\[\033[38;5;12m\]]\[$(tput sgr0)\]:\[$(tput sgr0)\]\[\033[38;5;12m\]\W\[$(tput sgr0)\]\\$ \[$(tput sgr0)\]\"" >> ~/.bashrc
fi

# Final Messages ################################################################
final_msg="${PRP} Setup complete!
====================================================================================================
 Instructions:
  Access VS Code enviroment:${WHT} http://${LOCAL_HOSTNAME}:${VSCODE_PORT_PRINT}/?folder=/workspace/${PRP}
  Access Jupyter Lab enviroment:${WHT} http://${LOCAL_HOSTNAME}:${JUPYTER_PORT_PRINT}/lab/tree/workspace/${PRP}
  Use ${NC}exit${PRP} to break and kill this container. ${YLW}WARNING${PRP} Save all code before doing this!
  Use ${NC}CRTL-p ${PRP}& ${NC}CTRL-q ${PRP}to detach from the container and while keeping it running
  Use ${NC}docker attach ${CONTAINER_ID} ${PRP}to re-attach to a running container shell
  Use ${NC}docker exec -it ${CONTAINER_ID} bash ${PRP}to attach to a new container shell
  Use ${NC}docker kill ${CONTAINER_ID} ${PRP}to kill a running container.
====================================================================================================${NC}"
echo -e "${final_msg}"

/bin/bash

# Break Container message #######################################################
end_msg="${PRP}====================================================================================================
 Closing ${GRN}${IMAGE_NAME} ${PRP}container for: ${GRN}${PROJECT_NAME}${PRP}
====================================================================================================${NC}"
echo -e "${end_msg}"
