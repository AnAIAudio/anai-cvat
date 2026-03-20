import base64
import io
import json

import yaml
from model_handler import ModelHandler
from PIL import Image


def init_context(context):
    context.logger.info("Init context...  0%")

    with open("/opt/nuclio/function.yaml", "rb") as function_file:
        functionconfig = yaml.safe_load(function_file)

    annotations = functionconfig["metadata"]["annotations"]
    labels_spec = annotations["spec"]
    labels = {item["id"]: item["name"] for item in json.loads(labels_spec)}

    yolo_labels_str = annotations.get("yolo_labels", "{}")
    yolo_labels = {int(k): v for k, v in json.loads(yolo_labels_str).items()}

    cluster_label_map_str = annotations.get("cluster_label_map", "{}")
    cluster_label_map = {int(k): v for k, v in json.loads(cluster_label_map_str).items()}

    kmeans_model_path = "/opt/nuclio/" + annotations.get("kmeans_model", "player_kmeans.joblib")
    onnx_model_path = "/opt/nuclio/" + annotations.get("onnx_model", "best_new.onnx")

    model = ModelHandler(yolo_labels, onnx_model_path, kmeans_model_path, cluster_label_map)
    context.user_data.model = model

    context.logger.info("Init context...100%")


def handler(context, event):
    context.logger.info("Run YOLO 11 Football ONNX model")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    threshold = float(data.get("threshold", 0.5))
    image = Image.open(buf).convert("RGB")

    results = context.user_data.model.infer(image, threshold)

    return context.Response(
        body=json.dumps(results), headers={}, content_type="application/json", status_code=200
    )
