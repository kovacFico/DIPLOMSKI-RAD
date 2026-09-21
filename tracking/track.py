import numpy as np
from tracking.color_model import ColorModel


class Track:
    '''
    Track predstavlja stanje pojedinačnog traga koji se prati kroz video. 
    
    Atributi Tracka su podjeljene na 4 cijeline i opsiuju:
    1. Identitet tragača (id i model boje)
    2. Trenutnu detekciju (bounding box, ključne točke, centar, značajka boje)
    3. Status tragača (aktivnost, broj propuštenih okvira)
    4. Appearance učenje (buffer značajki, interval ažuriranja, broj okvira od posljednjeg ažuriranja)
    
    Identitet nije vezan uz geometrijske značajke, već se temelji na naučenom modelu boje.
    
    Atributi:
    - id: Jedinstveni identifikator tragača.
    - color_model: Instanca ColorModel koja uči i procjenjuje boju rukavice.
    
    - bbox: Trenutni bounding box tragača (x1, y1, x2, y2).
    - keypoints: Trenutne ključne točke tragača (npr. zglobovi).
    - last_feature: Posljednja značajka korištena za usporedbu (npr. boja rukavice).
    - last_distance: Posljednja udaljenost između značajke i modela boje.
    - last_quality: Posljednja kvaliteta detekcije (npr. kombinacija udaljenosti i pokreta).
    - last_update_accepted: Boolean koji označava je li posljednja detekcija prihvaćena za ažuriranje modela.
    
    - missing_frames: Broj uzastopnih okvira u kojima tragač nije detektiran.
    - active: Boolean koji označava je li tragač trenutno aktivan
    
    - feature_buffer: Lista značajki prikupljenih za učenje modela boje.
    - update_interval: Broj okvira između ažuriranja modela boje.
    - frames_since_update: Broj okvira od posljednjeg ažuriranja modela
    '''

    def __init__(self, track_id):

        self.id = track_id
        self.color_model = ColorModel()

        self.bbox = None
        self.keypoints = None
        self.center = None
        self.last_feature = None
        self.last_distance = None
        self.last_quality = None
        self.last_update_accepted = False

        self.missing_frames = 0
        self.active = False

        self.feature_buffer = []
        self.update_interval = 5
        self.frames_since_update = 0