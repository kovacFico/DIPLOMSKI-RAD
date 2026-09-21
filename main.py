import cv2
import numpy as np
from ultralytics import YOLO

from detection.pose_detector import PoseDetector
from preprocessing.preprocessing import apply_clahe
from preprocessing.temporal import TemporalSmoother
from tracking.tracker import Tracker

from utils.visualization import (
    draw_debug_overlay,
    create_multiview_frame,
    resize_for_display
)
from utils.glove_extraction import (
    extract_glove_roi,
    extract_glove_features_lab,
    is_valid_glove_roi
)

from evaluation.metrics import (
    IdentityEvaluator,
    CoverageEvaluator
)
from evaluation.ground_truth_2_1 import (
    EXPECTED_IDENTITY,
    WRONG_IDENTITY_INTERVALS
)

from multiview.global_identity_resolver import GlobalIdentityResolver
from evaluation.multiview_metrics import MultiViewIdentityEvaluator
from evaluation.multiview_ground_truth_2 import (
    VIEW_GROUND_TRUTH,
    GLOBAL_EXPECTED_IDENTITY
)

from evaluation.print_view_metrics import (
    print_single_view_metrics,
    print_multiview_metrics
)



CONF_THRESHOLD = 0.4

VIDEO_PATH_SINGLE = "runda2-1.mp4" 
VIDEO_PATH_MULTI = [
    "runda2-0.mp4",
    "runda2-1.mp4"
]

MODE = 'multi'  # 'single' ili 'multi'
EVALUATE = True  
SHOW_DEBUG = True


def process_frame(frame, pose_detector):
    """
    Iz jednog framea gradi detections i debug_data.
    """

    detections = []
    debug_data = []

    detections_raw = pose_detector(frame)

    for det in detections_raw:
        if det["confidence"] < CONF_THRESHOLD:
            continue

        bbox = det["bbox"]
        keypoints = det["keypoints"]

        features = []

        feature_left = None
        feature_right = None

        feature_msg_l = "Feature extraction not attempted"
        feature_msg_r = "Feature extraction not attempted"

        roi_l, c_l = extract_glove_roi(frame, keypoints, "left", bbox)
        valid_l, msg_l = is_valid_glove_roi(roi_l)
        if valid_l:
            feature_left, feature_msg_l = extract_glove_features_lab(roi_l)
            if feature_left is not None:
                features.append(("left", feature_left))

        roi_r, c_r = extract_glove_roi(frame, keypoints, "right", bbox)
        valid_r, msg_r = is_valid_glove_roi(roi_r)
        if valid_r:
            feature_right, feature_msg_r = extract_glove_features_lab(roi_r)
            if feature_right is not None:
                features.append(("right", feature_right))

        # pokazuje ne samo "je li crop postojao", nego "je li taj 
        # crop stvarno ušao u appearance pipeline".
        usable_l = valid_l and feature_left is not None
        usable_r =  valid_r and feature_right is not None


        feature_side = None
        best_feature = None
        best_quality = np.inf

        for side, feature in features:
            quality = (feature[2] + feature[3])

            if quality < best_quality:
                best_quality = quality
                best_feature = feature
                feature_side = side


        detections.append({
            "bbox": bbox,
            "keypoints": keypoints,
            "feature": best_feature,
            "quality": best_quality
        })


        debug_data.append({
            "roi_left": roi_l,
            "center_left": c_l,
            "valid_left": usable_l,
            "msg_left": msg_l,
            "feature_left": feature_left,
            "feature_msg_left": feature_msg_l,

            "roi_right": roi_r,
            "center_right": c_r,
            "valid_right": usable_r,
            "msg_right": msg_r,
            "feature_right": feature_right,
            "feature_msg_right": feature_msg_r,

            "feature": best_feature,
            "quality": best_quality,
            "selected_side": feature_side
        })


    return detections, debug_data



def run_multi():
    '''
    Pokreće multi-view pipeline nad dvije sinkronizirane video snimke.

    Za svaki frame oba viewa:
        1. Izvršava preprocessing.
        2. Izvršava pose detekciju i ekstrakciju glove featurea.
        3. Ažurira dva nezavisna lokalna Trackera.
        4. GlobalIdentityResolver povezuje njihove lokalne identitete.
        5. MultiViewIdentityEvaluator ažurira evaluacijske metrike.
        6. Po potrebi se prikazuje multi-view debug vizualizacija.

    Obrada prestaje kada završi bilo koja od dvije snimke.
    '''
    
    cap0 = cv2.VideoCapture(VIDEO_PATH_MULTI[0])
    cap1 = cv2.VideoCapture(VIDEO_PATH_MULTI[1])

    if not cap0.isOpened():
        raise RuntimeError(
            "Cannot open view 0"
        )

    if not cap1.isOpened():
        raise RuntimeError(
            "Cannot open view 1"
        )

    model = YOLO("yolo11n-pose.pt")
    pose_detector = PoseDetector(model)

    tracker0 = Tracker(max_missing=30)
    tracker1 = Tracker(max_missing=30)

    smoother0 = TemporalSmoother(alpha=0.6)
    smoother1 = TemporalSmoother(alpha=0.6)


    multi_tracker = GlobalIdentityResolver(
        trackers=[tracker0, tracker1],
        min_model_updates=10,
        owner_timeout=15,
        switch_delay=10
    )

    if EVALUATE:
        fps = cap0.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
                
        multi_eval = MultiViewIdentityEvaluator(
            view_ground_truth=VIEW_GROUND_TRUTH,
            global_expected_identity=(
                GLOBAL_EXPECTED_IDENTITY
            ),
            fps=fps
        )


    frame_idx = 0
    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1:
            break
        
        frame0 = apply_clahe(frame0)
        frame1 = apply_clahe(frame1)

        frame0 = smoother0.apply(frame0)
        frame1 = smoother1.apply(frame1)

        detections0, debug0 = process_frame(frame0, pose_detector)
        detections1, debug1 = process_frame(frame1, pose_detector)

        tracker0.update(detections0)
        tracker1.update(detections1)
        multi_tracker.update()

        if EVALUATE:
            multi_eval.update(frame_idx, multi_tracker)

        if SHOW_DEBUG:
            frame0 = draw_debug_overlay(frame_idx, frame0, tracker0, debug0)
            frame1 = draw_debug_overlay(frame_idx, frame1, tracker1, debug1)

            frame = create_multiview_frame(frame0, frame1, multi_tracker.global_tracks)
            display_frame = resize_for_display(frame, max_width=1920, max_height=1080)
            cv2.imshow("Boxing Multi-View", display_frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        frame_idx += 1
        if frame_idx == 5100: break # radi evaluacije samo na prvih 5100 frameova, jer je to dovoljno za evaluaciju


    if EVALUATE:
        multi_eval.finalize()
        report = multi_eval.report()
        print_multiview_metrics(report)


    cap0.release()
    cap1.release()

    cv2.destroyAllWindows()
    
    

def run_single():
    '''
    Pokreće single-view tracking pipeline.

    Za svaki frame:
        1. Izvršava preprocessing.
        2. Izvršava pose detekciju i ekstrakciju glove featurea.
        3. Ažurira lokalni Tracker.
        4. Po potrebi ažurira evaluacijske metrike.
        5. Prikazuje debug vizualizaciju.

    Na kraju videa ispisuje rezultate single-view evaluacije.
    '''

    cap = cv2.VideoCapture(VIDEO_PATH_SINGLE)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    if EVALUATE:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
                
        id_eval = IdentityEvaluator(
            expected_identity=EXPECTED_IDENTITY,
            wrong_intervals=WRONG_IDENTITY_INTERVALS,
            fps=fps
        )
        cov_eval = CoverageEvaluator()

    model = YOLO("yolo11n-pose.pt")
    pose_detector = PoseDetector(model)
    
    tracker = Tracker(max_missing=30)
    smoother = TemporalSmoother(alpha=0.6)

    frame_idx = 0
    
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = apply_clahe(frame)
        frame = smoother.apply(frame)

        detections, debug_data = process_frame(frame, pose_detector)
        tracker.update(detections)

        if EVALUATE:
            id_eval.update(frame_idx, tracker.tracks)
            cov_eval.update(tracker.tracks)

        if SHOW_DEBUG:
            frame = draw_debug_overlay(frame_idx, frame, tracker, debug_data)
            display_frame = resize_for_display(frame, max_width=1920, max_height=1080)

            cv2.imshow("Boxing Single-View", display_frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        frame_idx += 1
        if frame_idx == 5100: break # radi evaluacije samo na prvih 5100 frameova, jer je to dovoljno za evaluaciju


    if EVALUATE:
        id_eval.finalize()
        id_report = id_eval.report()
        coverage_report = cov_eval.report()
        
        print_single_view_metrics(id_report, coverage_report)


    cap.release()
    cv2.destroyAllWindows()



def main():
    if MODE == "single":
        run_single()

    elif MODE == "multi":
        run_multi()

    else:
        raise ValueError(
            f"Unknown MODE: {MODE}"
        )



if __name__ == "__main__":
    main()
