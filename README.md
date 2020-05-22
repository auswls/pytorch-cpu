# pytorch-cpu

Pytorch CPU 버전 환경 구성을 위한 Dockerfile과 Deep learning training & prediction에 필요한 코드, data

## 사전 준비
weights 디렉토리에 학습 후 weight를 저장

## Docker Image & Container 생성 방법
<pre>
<code>
# Docker Image 생성
docker build -t {Docker image 이름} {Dockerfile 경로(현재 디렉토리인 경우 .)}

# Docker Container 생성
docker run -it \
      -v $(pwd)/weights:/workspace/weights \
      -v $(pwd)/flowers/test:/workspace/images \
      -v $(pwd)/prediction_result:/workspace/prediction_result \
      --name {Docker Container 이름} \
      {Docker Image 이름}
      
# Docker Container 생성 이후(run 이후)
docker start {Docker Contaier 이름}
docker attach {Docker Container 이름}
</code>
</pre>
