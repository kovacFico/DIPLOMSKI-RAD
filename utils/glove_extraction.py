import cv2
import numpy as np


def is_valid_glove_roi(roi, min_area=200):
    '''
    Minimalna validacija ROI-ja.
    
    ROI mora:
        - postojati
        - sadržavati piksele
        - imati dovoljno veliku površinu

    Args:
        roi (np.ndarray): ROI slike rukavice.
        min_area (int): Minimalan broj piksela.

    Returns:
        bool
    '''

    if roi is None:
        return False, "ROI is None"

    if roi.size == 0:
        return False, "ROI is empty"

    h, w = roi.shape[:2]
    area = h * w

    if area < min_area:
        return False, f"Area too small ({area} < {min_area})"

    return True, f"Valid ROI (area={area})"


def extract_glove_roi(
    frame,
    keypoints,
    side,
    bbox=None,
    scale_factor=0.6,
    min_size=20,
    roi_offset_ratio=0.30
):
    '''
    Izvlači ROI rukavice koristeći wrist keypoint kao bazu.
    ROI se dodatno pomiče u smjeru:
        elbow -> wrist
    kako bi centar ROI-a završio bliže stvarnom centru rukavice.

    Args:
        frame (np.ndarray): Okvir slike.
        keypoints (dict): Rječnik ključnih točaka s imenima i koordinatama.
        side (str): "left" ili "right": koju ruku analizirati.
        bbox (tuple | None): granični okvir detekcije (x1, y1, x2, y2) koji se koristi 
                            kao fallback za određivanje skale ROI-ja ako elbow keypoint nije dostupan.
        scale_factor (float): Koliki dio udaljenosti wrist-elbow koristiti kao baznu veličinu ROI-ja.
        min_size (int): Minimalna veličina ROI-ja u pikselima (po strani).
        roi_offset_ratio (float): Koliki dio ROI veličine koristiti za pomak centra prema rukavici.

    Returns:
        roi (np.ndarray | None): izrezani ROI slike rukavice ili None ako nije moguće izračunati.
        center (tuple | None): koordinate centra ROI-a (cx, cy) ili None ako ROI nije validan.
    '''

    wrist_key = f"{side}_wrist"
    elbow_key = f"{side}_elbow"

    if wrist_key not in keypoints:
        return None, None

    wx, wy = keypoints[wrist_key]
    if elbow_key in keypoints:
        ex, ey = keypoints[elbow_key]
        scale = np.linalg.norm([
            wx - ex,
            wy - ey
        ])
    elif bbox is not None:
        x1, y1, x2, y2 = bbox
        scale = 0.15 * (x2 - x1)
    else:
        scale = min_size

    size = max(int(scale * scale_factor), min_size)

    cx, cy = wx, wy
    if elbow_key in keypoints:
        direction = np.array([wx - ex, wy - ey])
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            direction /= norm
            cx = wx + direction[0] * size * roi_offset_ratio
            cy = wy + direction[1] * size * roi_offset_ratio


    x1 = int(cx - size)
    y1 = int(cy - size)

    x2 = int(cx + size)
    y2 = int(cy + size)

    h, w = frame.shape[:2]

    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None, None

    roi = frame[y1:y2, x1:x2]

    return roi, (cx, cy)


def extract_glove_features_lab(
    roi,
    min_saturation=20,
    trim_percent=10,
    hist_bins=16,
    peak_radius=1
):
    '''
    Izvlači robustan feature vector rukavice iz ROI-ja.

    Pipeline:
        1. ROI -> LAB konverzija
        2. saturation filtering
        3. trimming outliera
        4. 2D histogram nad A/B kanalima
        5. dominant peak extraction
        6. mean/std oko dominantnog peaka

    Args:
        roi (np.ndarray): ROI slike rukavice.
        min_saturation (int): Minimalna HSV saturacija za validne piksele.
        trim_percent (float): Percentile trimming za uklanjanje outliera.
        hist_bins (int): Broj histogram binova po kanalu.
        peak_radius (int): Koliko susjednih binova uključiti oko dominantnog peaka.

    Returns:
        np.ndarray or None: Feature vector oblika
            [mean_A, mean_B, std_A, std_B]
            ili None ako ekstrakcija nije uspjela.
    '''


    # ZA PROVJERITI I DORADITITI:
    # 1. min_saturation: možda treba prilagoditi ovisno o uvjetima osvjetljenja
    # 2. trim_percent: preveliki trim može ukloniti previše podataka, premali može ostaviti outliere4
    # 3. uvjet za mask.sum() < 10: možda treba povećati ili smanjiti ovisno o veličini ROI-a i očekivanom broju validnih piksela


    if roi is None or roi.size == 0:
        return (None, "ROI does not exist or is empty")

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lab_flat = lab.reshape(-1, 3).astype(np.float32)
    hsv_flat = hsv.reshape(-1, 3)


    saturation = hsv_flat[:, 1]
    mask = saturation > min_saturation

    # Možda premali uvjet, tj broj 10
    if mask.sum() < 10:
        return (None, f"Not enough valid pixels after saturation filtering: {mask.sum()} < 10")

    lab_valid = lab_flat[mask]
    

    lower = np.percentile(lab_valid, trim_percent, axis=0)
    upper = np.percentile(lab_valid, 100 - trim_percent, axis=0)
    trimmed = lab_valid[np.all((lab_valid >= lower) & (lab_valid <= upper), axis=1)]
    if trimmed.shape[0] < 10: 
        return (None, f"Not enough valid pixels after trimming: {trimmed.shape[0]} < 10")
    

    ab = trimmed[:, 1:3]
    hist, a_edges, b_edges = np.histogram2d(
        ab[:, 0],
        ab[:, 1],
        bins=hist_bins,
        range=[[0, 255], [0, 255]]
    )


    peak_idx = np.unravel_index(np.argmax(hist), hist.shape)
    peak_a_idx, peak_b_idx = peak_idx

    a_min_idx = max(0, peak_a_idx - peak_radius)
    a_max_idx = min(hist_bins - 1, peak_a_idx + peak_radius)

    b_min_idx = max(0, peak_b_idx - peak_radius)
    b_max_idx = min(hist_bins - 1, peak_b_idx + peak_radius)

    a_min = a_edges[a_min_idx]
    a_max = a_edges[a_max_idx + 1]

    b_min = b_edges[b_min_idx]
    b_max = b_edges[b_max_idx + 1]

    peak_mask = (
        (ab[:, 0] >= a_min) &
        (ab[:, 0] < a_max) &
        (ab[:, 1] >= b_min) &
        (ab[:, 1] < b_max)
    )

    peak_pixels = ab[peak_mask]
    if peak_pixels.shape[0] < 20:  # Možda treba prilagoditi ovaj uvjet ovisno o veličini ROI-a i očekivanom broju pikseli u dominantnom clusteru
        return (None, f"Not enough valid pixels in dominant cluster: {peak_pixels.shape[0]} < 20")


    mean_ab = np.mean(peak_pixels, axis=0)
    std_ab = np.std(peak_pixels, axis=0)

    feature = np.array([
        mean_ab[0],
        mean_ab[1],
        std_ab[0],
        std_ab[1]
    ], dtype=np.float32)
    
    quality = std_ab[0] + std_ab[1]

    return (feature, (f"Feature accepted | "
                      f"peak_pixels={peak_pixels.shape[0]}, "
                      f"Quality={quality:.2f}"))


