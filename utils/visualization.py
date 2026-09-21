import cv2
import numpy as np
from typing import Dict, List
from tracking.track import Track
from tracking.tracker import Tracker


def get_track_color(track: Track):
    """
    Vraća približnu BGR boju Color modela koristeći
    srednje A/B vrijednosti i fiksnu L komponentu.
    """

    cm = track.color_model

    if not cm.is_initialized or cm.mean is None:
        return (150, 150, 150)

    mean_a = cm.mean[0]
    mean_b = cm.mean[1]

    lab = np.uint8([[[180, mean_a, mean_b]]])
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)[0, 0]

    return tuple(int(v) for v in bgr)


def draw_color_patch(frame, bgr_color, position):
    """
    Crta kvadrat zadane BGR boje.
    """

    if bgr_color is None:
        return

    x, y = position

    cv2.rectangle(
        frame,
        (x, y),
        (x + 20, y + 20),
        bgr_color,
        -1
    )

    cv2.rectangle(
        frame,
        (x, y),
        (x + 20, y + 20),
        (255,255,255),
        1
    )


def draw_text(
    frame,
    text,
    position,
    color,
    scale=0.45,
    thickness=1
):
    """
    Crta tekst s crnim obrubom radi bolje čitljivosti.
    """

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )
    
    
def draw_separator(frame, x, y, width):
    '''
    Crta horizontalnu liniju separatora.
    '''
    
    cv2.line(
        frame,
        (x,y),
        (x+width,y),
        (120,120,120),
        1
    )
    
    
def draw_field(frame, label, value, x, y, color):
    """
    Crta jednu stavku u obliku:
        LABEL : value
    """
    draw_text(
        frame,
        f"{label:<8}: {value}",
        (x, y),
        color
    )

    
def draw_track(frame, track: Track, color):
    """
    Crta bounding box i ID tragača na okviru. Također prikazuje status tragača
    (ACTIVE ili MISSING s brojem propuštenih okvira).
    
    Args:
        frame (np.ndarray): Okvir slike.
        track (Track): Objekt tragača.
        color (tuple): BGR boja za crtanje.
        
    Returns:
        np.ndarray: Okvir sa iscrtanom informacijom o tragaču.
    """
    
    if track.bbox is None:
        return frame

    x1, y1, x2, y2 = map(int, track.bbox)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    status = "ACTIVE" if not track.missing_frames else f"MISSING ({track.missing_frames})"
    label = f"ID {track.id} | {status}"

    draw_text(
        frame,
        label,
        (x1, y1 - 10),
        color,
        scale=0.5,
        thickness=2
    )
    
    return frame


def draw_keypoints(frame, keypoints, color):
    """
    Crta ključne točke (keypoints) na okviru slike.
    
    Args:
        frame (np.ndarray): Okvir slike.
        keypoints (dict): Rječnik ključnih točaka s imenima i koordinatama.
        color (tuple): BGR boja za crtanje.
        
    Returns:
        np.ndarray: Okvir sa iscrtanim ključnim točkama.
    """
    
    for name, (x, y) in keypoints.items():
        if "wrist" in name:
            cv2.circle(frame, (int(x), int(y)), 4, color, -1)

    return frame


def draw_glove_roi(frame, roi, center, valid, side=None, selected=False):
    """ 
    Crta pravokutnik oko ROI-a rukavice na okviru slike.
    Boje:
        zelena - ROI je usable i odabran
        žuta   - ROI je usable, ali nije odabran
        crvena - ROI nije usable i nije odabran
       
    
    Args:
        frame (np.ndarray): Okvir slike.
        roi (np.ndarray): ROI slike rukavice.
        center (tuple): Centar ROI-a kao (x, y).
        valid (bool): Označava je li ROI valjan.
        side (str, optional): Strana rukavice ('left' ili 'right').
        selected (bool): Označava je li ROI odabran.
        
    Returns:
        np.ndarray: Okvir sa iscrtanim ROI-jem rukavice.
    """
    
    if roi is None or center is None:
        return frame

    cx, cy = map(int, center)
    h, w = roi.shape[:2]
    half_w = w // 2
    half_h = h // 2

    x1 = cx - half_w
    y1 = cy - half_h
    x2 = cx + half_w
    y2 = cy + half_h

    if selected:
        color = (0, 255, 0)        # green
        thickness = 3

    elif valid:
        color = (0, 255, 255)      # yellow
        thickness = 2

    else:
        color = (0, 0, 255)        # red
        thickness = 2

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    if side is not None:
        draw_text(
            frame,
            side.upper(),
            (x1, y1 - 5),
            color,
            scale=0.4,
            thickness=1
        )
        
    return frame


def draw_track_panel(frame_idx,frame, track: Track, origin, color):
    """
    Crta telemetrijske informacije o tragu borca na okviru slike.
    
    Args:
        frame (np.ndarray): Okvir slike.
        track (Track): Objekt traga borca.
        origin (tuple): Gornji lijevi kut panela kao (x, y).
        color (tuple): BGR boja za tekst.
        
    Returns:
        np.ndarray: Okvir sa ispisanim informacijama o tragaču.
    """
    x, y = origin
    cm = track.color_model
    
    # color = (255, 255, 255) # za tekst fiksiran u bijelo, a ne u boji Tracka
    
    draw_text(frame, f"TRACK {track.id}: {frame_idx}", (x, y), color)
    draw_separator(frame, x, y + 8, 170)
    y += 25

    draw_text(frame, "STATUS", (x, y), color)
    y += 18
    draw_field(frame, "Active", track.active, x, y, color)
    y += 18
    draw_field(frame, "Missing", track.missing_frames, x, y, color)

    if cm.mean is not None:
        mean_a = cm.mean[0]
        mean_b = cm.mean[1]
        y += 25
        draw_separator(frame, x, y, 170)
        y += 18
        draw_text(frame, "MODEL", (x, y), color)
        y += 18
        draw_color_patch(frame, get_track_color(track), (x, y))
        y += 40
        draw_field(frame, "A", f"{mean_a:.1f}", x, y, color)
        y += 18
        draw_field(frame, "B", f"{mean_b:.1f}", x, y, color)

    if track.last_feature is not None:
        a = track.last_feature[0]
        b = track.last_feature[1]

        std_a = track.last_feature[2]
        std_b = track.last_feature[3]

        quality = std_a + std_b
        
        y += 25
        draw_separator(frame, x, y, 170)
        y += 18
        draw_text(frame, "FEATURE", (x, y), color)
        y += 25
        draw_field(frame, "A", f"{a:.1f}", x, y, color)
        y += 18
        draw_field(frame, "B", f"{b:.1f}", x, y, color)
        y += 18
        draw_field(frame, "STD A", f"{std_a:.1f}", x, y, color)
        y += 18
        draw_field(frame, "STD B", f"{std_b:.1f}", x, y, color)
        y += 18
        draw_field(frame, "Q", f"{quality:.1f}", x, y, color)
        
    if track.last_distance is not None:
        y += 25
        draw_separator(frame, x, y, 170)
        y += 18
        draw_text(frame, "DISTANCE", (x, y), color)
        y += 18
        draw_field(frame, "Dist", f"{track.last_distance:.2f}", x, y, color)
        
    y += 25
    draw_separator(frame, x, y, 170)
    y += 18
    draw_text(frame, "LEARNING", (x, y), color)
    y += 18
    draw_field(frame, "Buffer", f"{len(track.feature_buffer)}/{track.update_interval}", x, y, color)

    return frame
    

def draw_debug_overlay(frame_idx, frame, tracker: Tracker, debug_data: List[Dict] = None):
    """
    Ispisuje kompletan debug prikaz na okviru slike, uključujući bounding boxove, ključne točke, 
    ROI rukavica i panele s informacijama o pojedinom tragu borca.
    
    Args:
        frame (np.ndarray): Okvir slike.
        tracker (Tracker): Objekt tragača koji sadrži sve tragove i njihove informacije.
        debug_data (List[Dict], optional): Podaci za debugiranje.
        
    Returns:
        np.ndarray: Okvir sa svim iscrtanim informacijama.
    """
    if debug_data is not None:
        for data in debug_data:
            selected_side = data["selected_side"]
            draw_glove_roi(
                frame,
                data["roi_left"],
                data["center_left"],
                data["valid_left"],
                side="L",
                selected=(selected_side == "left")
            )
            draw_glove_roi(
                frame,
                data["roi_right"],
                data["center_right"],
                data["valid_right"],
                side="R",
                selected=(selected_side == "right")
            )
            
    colors = {
        track.id: get_track_color(track)
        for track in tracker.tracks
    }    
    
    for track in tracker.tracks:
        color = colors[track.id]
        draw_track(frame, track, color)
        
        if track.keypoints is not None:
            draw_keypoints(frame, track.keypoints, color)


    draw_track_panel(frame_idx,frame, tracker.tracks[0], (20, 30), colors[0])
    draw_track_panel(frame_idx, frame, tracker.tracks[1], (frame.shape[1] - 240, 30), colors[1])
    
    return frame
            

def create_multiview_frame(frame0, frame1, global_tracks, panel_height=45):
    '''
    Spaja debug prikaze iz dva viewa u jedan zajednički multi-view frame.

    Oba framea se prvo skaliraju na jednaku visinu uz očuvanje omjera stranica.
    Nakon toga se spajaju horizontalno. Na vrhu zajedničkog prikaza ispisuje se
    trenutno stanje svakog globalnog identiteta, odnosno iz kojeg viewa i kojeg
    lokalnog Tracka trenutno koristi podatke.

    Args:
        frame0 (np.ndarray): Debug frame prvog viewa.
        frame1 (np.ndarray): Debug frame drugog viewa.
        global_tracks (list[GlobalTrack]): Lista globalnih identiteta.
        panel_height (int): Visina donjeg status panela.


    Returns:
        np.ndarray: Zajednički multi-view frame.
    '''

    target_height = min(
        frame0.shape[0],
        frame1.shape[0]
    )

    h0, w0 = frame0.shape[:2]
    h1, w1 = frame1.shape[:2]

    scale = target_height / h0
    new_width0 = int(w0 * scale)

    scale = target_height / h1
    new_width1 = int(w1 * scale)

    frame0 = cv2.resize(frame0, (new_width0, target_height))
    frame1 = cv2.resize(frame1, (new_width1, target_height))

    combined = cv2.hconcat([frame0, frame1])

    draw_text(
        combined,
        "VIEW 0",
        (20, 25),
        (255, 255, 255),
        scale=0.6,
        thickness=2
    )

    draw_text(
        combined,
        "VIEW 1",
        (frame0.shape[1] + 20, 25),
        (255, 255, 255),
        scale=0.6,
        thickness=2
    )

    panel = np.zeros((
            panel_height,
            combined.shape[1],
            3
        ), dtype=np.uint8
    )

    y = 30
    for global_track in global_tracks:
        if global_track.now_using == "primary":
            assignment = global_track.primary
        elif global_track.now_using == "secondary":
            assignment = global_track.secondary
        else:
            assignment = None


        if assignment is None:
            text = (
                f"GLOBAL {global_track.global_id} | "
                f"NO ACTIVE OWNER"
            )
        else:
            text = (
                f"GLOBAL {global_track.global_id} | "
                f"VIEW {assignment.view_id} | "
                f"TRACK {assignment.track.id} | "
                f"{global_track.now_using.upper()}"
            )

        draw_text(
            combined,
            text,
            (20, combined.shape[0] - 80 + y),
            (255, 255, 255),
            scale=0.55,
            thickness=2
        )
        
        y += 28
        
    combined = cv2.vconcat([combined, panel])
    return combined


def resize_for_display(frame, max_width=1920, max_height=1080):
    '''
    Skalira frame tako da stane unutar zadane maksimalne rezolucije
    uz očuvanje originalnog omjera stranica.

    Frame se smanjuje samo ako prelazi maksimalnu širinu ili visinu.

    Args:
        frame (np.ndarray): Frame koji se prikazuje.
        max_width (int): Maksimalna širina prikaza.
        max_height (int): Maksimalna visina prikaza.

    Returns:
        np.ndarray: Skalirani frame spreman za prikaz.
    '''

    h, w = frame.shape[:2]

    scale = min(
        max_width / w,
        max_height / h,
        1.0
    )

    if scale == 1.0:
        return frame

    new_width = int(w * scale)
    new_height = int(h * scale)

    return cv2.resize(
        frame,
        (new_width, new_height)
    )