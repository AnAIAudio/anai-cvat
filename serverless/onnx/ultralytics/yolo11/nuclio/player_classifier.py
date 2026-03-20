import cv2
import numpy as np
import joblib


RESIZE = (32, 64)  # (width, height) — 학습 시 사용한 크기와 동일


class PlayerClassifier:
    def __init__(self, kmeans_model_path, cluster_label_map, default_label="player"):
        self.global_kmeans = joblib.load(kmeans_model_path)
        self.cluster_label_map = cluster_label_map
        self.default_label = default_label

    def classify(self, frame, bbox):
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        crop = frame[y1:y2, x1:x2]

        if crop.shape[0] < 4 or crop.shape[1] < 4:
            return self.default_label

        try:
            resized = cv2.resize(crop, RESIZE)
            feature = resized.flatten().astype(np.float32).reshape(1, -1)
            cluster_id = self.global_kmeans.predict(feature)[0]
            return self.cluster_label_map.get(cluster_id, self.default_label)
        except Exception:
            return self.default_label
