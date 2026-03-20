FROM ubuntu:22.04

RUN apt-get update && \
    apt-get install --no-install-recommends -y python3-pip python-is-python3 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install onnxruntime opencv-python-headless pillow pyyaml numpy scikit-learn joblib --no-cache-dir

WORKDIR /opt/nuclio

COPY best_new.onnx /opt/nuclio/best_new.onnx
COPY player_kmeans.joblib /opt/nuclio/player_kmeans.joblib
