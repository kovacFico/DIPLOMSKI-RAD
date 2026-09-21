import numpy as np

class PoseDetector:
    def __init__(self, model):
        """
        Inicijalizira detektor poza s danim modelom.
        
        Args:
            model: Model za detekciju poza (npr. YOLOv8 pose model).
        """
        self.model = model

    def __call__(self, frame):
        """
        Izvršava detekciju poza na danom okviru.
        
        Args:
            frame (np.ndarray): Okvir slike na kojem se vrši detekcija.
            
        Returns:
            list: Lista detekcija, gdje je svaka detekcija dict s ključevima:
                  - "bbox": tuple (x1, y1, x2, y2)
                  - "confidence": float
                  - "keypoints": dict s ključevima za zglobove i njihovim (x, y) koordinatama
        """
        results = self.model(frame, verbose=False)
        detections = []

        for r in results:
            if r.boxes is None or r.keypoints is None:
                continue

            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            kps = r.keypoints.xy.cpu().numpy()

            for bbox, score, kp in zip(boxes, scores, kps):
                keypoints = self._parse_keypoints(kp)

                detections.append({
                    "bbox": tuple(bbox),
                    "confidence": float(score),
                    "keypoints": keypoints
                })

        return detections

    def _parse_keypoints(self, kp_array):
        """
        Parsira niz ključnih točaka u Dictionary s imenima zglobova.
        """
        
        # POTREBNO KASNIJE NADODATI KEYPOINTOVE ZA GLAVU RADI HIT DETEKCIJE 
        return {
            "left_wrist": tuple(kp_array[9]),
            "right_wrist": tuple(kp_array[10]),
            "left_elbow": tuple(kp_array[7]),
            "right_elbow": tuple(kp_array[8]),
            "left_shoulder": tuple(kp_array[5]),
            "right_shoulder": tuple(kp_array[6])
        }
