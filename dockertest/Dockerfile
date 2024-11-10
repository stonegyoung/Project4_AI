# https://fastapi.tiangolo.com/ko/deployment/docker/#_6
# 3.10 버전 다운
FROM python:3.10

# requirements.txt 파일과 app 디렉터리를 위치
WORKDIR /code

# code 디렉토리에 복사
COPY . /code

# 라이브러리 설치
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --no-deps -r requirements.txt


# CMD ["gunicorn", "--bind", "0.0.0.0:9200", "main:app", "--workers", "5", "--worker-class", "uvicorn.workers.UvicornWorker"]
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9100", "--workers", "5"]

# requirements.txt에서 pywin32랑 pywinpty 지우기
# /code로 이동한 후 이미지 생성 docker build -t unilogtest .
# 도커 컨테이너 시작 docker run -it --name uni1106 -p 9100:9100  -v /C:/project4/dockerlog:/code/logs unilogtest
