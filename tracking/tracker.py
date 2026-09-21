import numpy as np
from scipy.optimize import linear_sum_assignment
from tracking.track import Track

# POSITION GATE JOŠ NIJE IMPLEMENTIRAN, ALI SE MOŽE DODATI KAO PRE-PRIJEVREMENA PROVJERA PRIJE RAČUNANJA COST MATRICE
'''raw_distance = np.linalg.norm(...)

if raw_distance > self.max_position_distance:
    continue

position_cost = (
    raw_distance /
    self.max_position_distance
)'''

class Tracker:
    '''
    Prilagođeni Tracker za dva objekta(borca).
    
    Koristi:
        - Poziciju (Euclidean distance)
        - Boju (Online EMA model boje s Mahalanobis udaljenosti) 
        
    Matching se vrši pomoću Hungarian algoritma. Matrica je konstruirana tako da
    se pozicijski trošak normalizira ovisno o maksimalnoj očekivanoj udaljenosti,
    a trošak boje se računa pomoću Mahalanobisove udaljenosti između nove značajke 
    i modela boje tragača. Ako detekcija nema valjanu značajku boje, koristi se 
    unaprijed definirana kazna. Težine za poziciju i boju su podešene empirijski, 
    a moguće ih je dodatno optimizirati.
    
    Args:
        max_missing (int): Maksimalan broj uzastopnih okvira u kojima 
                        traženi objekt može biti neaktivan prije nego 
                        što se označi kao neaktivan.   
        color_missing_penalty (float): Kazna za nedostajuću boju u cost funkciji.
        w_pos (float): Težina pozicijskog troška u ukupnom costu.
        w_color (float): Težina troška boje u ukupnom costu
    '''
    
    def __init__(self, max_missing=30):
        self.tracks = [Track(0), Track(1)]
        self.max_missing = max_missing
        self.color_missing_penalty = 0.125
        self.max_position_distance = 200
        self.w_pos = 1.2
        self.w_color = 1.5
        
        
    def bbox_center(self, bbox): 
        '''
        Računa centar bounding boxa. 
        
        Args:
            bbox (tuple): Bounding box u formatu (x1, y1, x2, y2).
            
        Returns:
            np.ndarray: Koordinate centra (x, y).
        '''
        x1, y1, x2, y2 = bbox 
        
        return np.array([
            (x1 + x2) / 2, 
            (y1 + y2) / 2
        ])
    
    
    def position_weight(self, track: Track):
        '''
        Dinamički prilagođava težinu pozicijskog troška na temelju broja propuštenih okvira.
        
        Args:
            track (Track): Tragač za koji se računa težina.
            
        Returns:
            float: Težina pozicijskog troška.
        '''

        m = track.missing_frames

        if m <= 10: return 1.2
        if m <= 20: return 0.5

        return 0.1
        

    def compute_cost_matrix(self, detections):
        '''
        Računa cost matricu između trenutnih Trackova i novih detekcija. Matrica se konstruira tako da 
        se pozicijski trošak normalizira ovisno o maksimalnoj očekivanoj udaljenosti, 
        a trošak boje se računa pomoću Mahalanobisove udaljenosti između nove značajke i modela boje 
        tragača. Ako detekcija nema valjanu značajku boje, koristi se unaprijed definirana kazna.
        
        Args:
            detections (list): Lista detekcija, svaka detekcija je dict s
                               ključima "position", "lab_color", "bbox", "keypoints".
        
        Returns:
            np.ndarray: Cost matrica oblika (num_tracks, num_detections).
        '''
        num_dets = len(detections)
        cost = np.full((len(self.tracks), num_dets), 1e6)

        for t_id, track in enumerate(self.tracks):
            track_center = None
            
            if track.bbox is not None:
                track_center = self.bbox_center(track.bbox)
                
            for d_id, det in enumerate(detections):
                position_cost = 0.0
                if track_center is not None:
                    det_center = self.bbox_center(det["bbox"])
                    position_cost = (
                        np.linalg.norm(det_center - track_center) /
                        self.max_position_distance
                    )
                
                feature = det["feature"]
                if not track.color_model.is_initialized:
                    color_cost = 0.0

                elif feature is None:
                    color_cost = self.color_missing_penalty

                else:
                    color_cost = track.color_model.distance(feature)
                
                w_pos = self.position_weight(track)

                cost[t_id, d_id] = (
                    w_pos * position_cost
                    +
                    self.w_color * color_cost
                )
                  
        return cost


    def update_apperance_model(self, track: Track):
        '''Ažurira model boje tragača na temelju prikupljenih značajki. Koristi se najbolja
        značajka iz buffer-a, a zatim se buffer čisti.
        
        Args:
            track (Track): Tragač čiji se model boje ažurira.
        '''
        
        if len(track.feature_buffer) == 0:
            return
        
        best_feature = min(track.feature_buffer, key=lambda f: f[2] + f[3])
        track.color_model.update(best_feature)
        track.feature_buffer.clear()
        track.frames_since_update = 0
        
        # ovdje bi bilo dobro napraviti opciju ako nijedan od 5 fetura ne zadovolji distancu 
        # samo kreni puniti ponovno od zadnjeg po principu FIFO, i tek kada se zadovolji distanca 
        # onda update
        # TAKOĐER možda napraviti još elegantnije, povećati buffer na 20 i kada u bufferu budemo
        # imali prvih 5 koji zadovoljavaju distancu, uzetio od njih prosjek i update, 
        # a buffer se čisti i puni ponovno. Također, ako se u bufferu ne nađe 5 koji zadovoljavaju 
        # distancu, onda se buffer puni opet po FIFO principu i kada se nađe 5 koji zadovoljavaju 
        # distancu, uzme se prosjek i update. time se uzima prosjek, što je uvijek najbolja opcija,
        # a ne samo najbolja značajka, jer se time smanjuje šum i varijabilnost boje, te se opet
        # čuva princip gdje se učenje odvija tek kada se zadovolji distanca, a ne samo kada se nađe najbolja značajka.


    def update(self, detections):
        '''
        Ažurira Trackove na temelju novih detekcija. Prvo se računa cost matrica između Trackova i 
        detekcija, zatim se koristi Hungarian algoritam za optimalno dodjeljivanje. Nakon dodjeljivanja, 
        Trackovi se ažuriraju sa novim informacijama, a neaktivni se označavaju nakon određenog 
        broja propuštenih okvira.
        
        Args:
            detections (list): Lista detekcija, svaka detekcija je dict s
                               ključima "position", "lab_color", "bbox", "keypoints".
        '''
        if len(detections) == 0:
            for track in self.tracks:
                track.missing_frames += 1
                if track.missing_frames > self.max_missing:
                    track.active = False
            return

        cost = self.compute_cost_matrix(detections)
        row_ind, col_ind = linear_sum_assignment(cost)

        assigned_tracks = set()

        for t_id, d_id in zip(row_ind, col_ind):
            if cost[t_id, d_id] >= 1e6:
                continue

            track = self.tracks[t_id]
            det = detections[d_id]
            
            track.bbox = det["bbox"]
            track.keypoints = det["keypoints"]
            track.last_quality = det["quality"]
            track.active = True
            track.missing_frames = 0
            
            feature = det["feature"]
            track.last_feature = feature
            
            
            if feature is not None:  
                if track.color_model.is_initialized:
                    track.last_distance = track.color_model.distance(feature)
            
                track.feature_buffer.append(feature)
                track.frames_since_update += 1
            else:
                track.last_distance = None
                
            if track.frames_since_update >= track.update_interval:
                self.update_apperance_model(track)
            
            assigned_tracks.add(t_id)
            
            
        for track in self.tracks:
            if track.id not in assigned_tracks:
                track.missing_frames += 1
                if track.missing_frames > self.max_missing:
                    track.active = False
