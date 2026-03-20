import cv2
import numpy as np
import onnxruntime as ort
from player_classifier import PlayerClassifier


class ModelHandler:
    def __init__(self, labels, onnx_model_path, kmeans_model_path, cluster_label_map):
        self.labels = labels
        self.model = None
        self.input_size = 640
        self.player_classifier = PlayerClassifier(kmeans_model_path, cluster_label_map)
        self.load_network(model=onnx_model_path)

    def load_network(self, model):
        device = ort.get_device()
        cuda = True if device == "GPU" else False
        try:
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if cuda
                else ["CPUExecutionProvider"]
            )
            so = ort.SessionOptions()
            so.log_severity_level = 3

            self.model = ort.InferenceSession(model, providers=providers, sess_options=so)
            self.output_details = [i.name for i in self.model.get_outputs()]
            self.input_details = [i.name for i in self.model.get_inputs()]
        except Exception as e:
            raise Exception(f"Cannot load model {model}: {e}")

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleup=True, stride=32):
        shape = im.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return im, r, (dw, dh)

    def infer(self, image, threshold):
        img = np.array(image)
        img_bgr = img[:, :, ::-1].copy()
        h, w, _ = img_bgr.shape

        # Preprocess
        img_lb, ratio, (dw, dh) = self.letterbox(img_bgr, new_shape=(self.input_size, self.input_size))
        img_lb = img_lb[:, :, ::-1].transpose((2, 0, 1))  # BGR to RGB, HWC to CHW
        img_lb = np.ascontiguousarray(img_lb, dtype=np.float32) / 255.0
        img_lb = np.expand_dims(img_lb, 0)

        # Run inference
        inp = {self.input_details[0]: img_lb}
        output = self.model.run(self.output_details, inp)[0]  # (1, 6, 8400)

        # Postprocess: YOLOv8/11 output format is (1, 4+num_classes, num_boxes)
        # Transpose to (num_boxes, 4+num_classes)
        output = output[0].T  # (8400, 6)

        boxes = output[:, :4]      # cx, cy, w, h
        scores = output[:, 4:]     # class scores (2 classes: ball, player)

        # Get best class per box
        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]

        # Filter by threshold
        mask = confidences >= threshold
        boxes = boxes[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]

        # Convert cx, cy, w, h -> x1, y1, x2, y2
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2

        # Rescale to original image size
        x1 = (x1 - dw) / ratio
        y1 = (y1 - dh) / ratio
        x2 = (x2 - dw) / ratio
        y2 = (y2 - dh) / ratio

        # NMS
        indices = cv2.dnn.NMSBoxes(
            bboxes=np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist(),
            scores=confidences.tolist(),
            score_threshold=threshold,
            nms_threshold=0.45,
        )

        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                raw_label = self.labels.get(int(class_ids[i]), "unknown")
                bbox = (
                    max(int(x1[i]), 0),
                    max(int(y1[i]), 0),
                    min(int(x2[i]), w),
                    min(int(y2[i]), h),
                )

                if raw_label == "player":
                    label = self.player_classifier.classify(img_bgr, bbox)
                else:
                    label = raw_label

                results.append({
                    "confidence": str(float(confidences[i])),
                    "label": label,
                    "points": list(bbox),
                    "type": "rectangle",
                })

        return results
