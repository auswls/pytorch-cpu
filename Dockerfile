# Docker for prediction (Ubuntu 18.04기반, pytorch cpu버전 이용)


###### Command for building and running docker ######
### Docker Image 생성 ###
# docker build -t {Docker image 이름} {Dockerfile 경로(현재 디렉토리인 경우 .)}

### Docker Container 생성 ###
# docker run -it \
#       -v $(pwd)/weights:/workspace/weights \
#       -v $(pwd)/flowers/test:/workspace/images \
#       -v $(pwd)/prediction_result:/workspace/prediction_result \
#       --name {Docker Container 이름} \
#       {Docker Image 이름}

### Docker Image 생성 이후(run 이후) ###
# docker start {Docker Contaier 이름}
# docker attach {Docker Container 이름}
######################################################

FROM ubuntu:18.04

# Install basic utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         git \
         curl \
         ca-certificates \
         libjpeg-dev \
         libpng-dev \
         libgtk2.0-dev && \
     rm -rf /var/lib/apt/lists/*

# Install Miniconda3 & Python 3.7
RUN curl -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

RUN  /opt/conda/bin/conda install conda-build && \
     /opt/conda/bin/conda create -y --name pytorch-py37 python=3.7.6 numpy pyyaml scipy ipython mkl&& \
     /opt/conda/bin/conda clean -ya

ENV PATH=/opt/conda/envs/pytorch-py37/bin:$PATH

RUN conda install --name pytorch-py37 pytorch torchvision -c soumith && /opt/conda/bin/conda clean -ya

# Create working directory
WORKDIR /workspace
RUN chmod -R a+w /workspace

# Copy codes from local
COPY src/ /workspace/src/
COPY flower_to_name.json /workspace/flower_to_name.json

# Install some python methods
RUN python -m pip install --upgrade pip && \
        pip install pandas && \
        python -m pip install opencv-python

# Set the default command to python3 (run, attach 시 실행)
CMD ["python", "src/test.py"]