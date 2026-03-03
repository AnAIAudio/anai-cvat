FROM ubuntu:22.04

RUN apt-get update && \
    apt-get install --no-install-recommends -y python3-pip python-is-python3 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install onnxruntime opencv-python-headless pillow pyyaml numpy --no-cache-dir

WORKDIR /opt/nuclio

COPY best1010_2.onnx /opt/nuclio/best1010_2.onnx
