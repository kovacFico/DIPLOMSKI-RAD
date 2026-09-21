import cv2
import numpy as np

def apply_clahe(frame, clip_limit=0.1, tile_grid_size=(4, 4)): #clip limit bio 2
    """
    Primjenjuje CLAHE (Contrast Limited Adaptive Histogram Equalization) na ulazni okvir slike
    kako bi se poboljšao kontrast i smanjila osvijetljenost, te tako poboljšala vidljivost rukavica.
    
    Args:
        frame (np.ndarray): Ulazni okvir slike u BGR formatu.
        clip_limit (float): Ograničenje za kontrast u CLAHE algoritmu.
        tile_grid_size (tuple): Veličina mreže pločica za CLAHE.
        
    Returns:
        np.ndarray: Obradjeni okvir slike u BGR formatu.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )
    l_eq = clahe.apply(l)

    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


