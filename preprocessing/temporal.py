import cv2

class TemporalSmoother:
    '''
    Primjenjuje vremensko izravnavanje na niz okvira slike. Na svaki novi okvir
    primjenjuje eksponencijalno izravnavanje koristeći zadani faktor alpha.
    
    Args:
        alpha (float): Faktor izravnavanja između 0 i 1. Više vrijednosti daju veći
                       značaj novim okvirima.
    
    Returns:
        np.ndarray: Izravnani okvir slike.
    '''
    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.prev = None

    def apply(self, frame):
        if self.prev is None:
            self.prev = frame.copy()
            return frame

        smoothed = cv2.addWeighted(
            frame, self.alpha,
            self.prev, 1 - self.alpha,
            0
        )
        self.prev = smoothed.copy()
        return smoothed
