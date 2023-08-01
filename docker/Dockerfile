FROM nvcr.io/nvidia/pytorch:23.04-py3
LABEL org.opencontainers.image.authors="soulteary@gmail.com"

RUN pip install gradio==3.39.0 lpips==0.1.4

WORKDIR /app
RUN git clone https://github.com/sczhou/CodeFormer.git && \
    cd CodeFormer && \
    git checkout 8392d0334956108ab53d9439c4b9fc9c4af0d66d

WORKDIR /app/CodeFormer/
COPY assets ./assets
COPY src/*.py ./
RUN code-fix.py
CMD ["python", "app.py"]