FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel

WORKDIR /workspace

RUN apt update && apt install -y git ffmpeg colmap

RUN git clone https://github.com/graphdeco-inria/gaussian-splatting.git

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY handler.py .

CMD ["python3", "-u", "handler.py"]
