import numpy as np
from scipy.optimize import linear_sum_assignment

from multiview.view_assignment import ViewAssignment
from multiview.global_track import GlobalTrack
from tracking.tracker import Tracker

class GlobalIdentityResolver:
    '''
    Klasa koja definira globalno povezivanje lokalnih Trackova iz dva viewa u jedan globalni identitet, 
    te donosi odluku kojem lokalnom Track objektu trenutno vjerujemo.
    '''
    

    def __init__(
        self, 
        trackers: list[Tracker],
        min_model_updates = 10,
        owner_timeout = 15,
        switch_delay = 10
    ):
        self.trackers = trackers
        self.min_model_updates = min_model_updates
        self.owner_timeout = owner_timeout
        self.switch_delay = switch_delay
        self.initialized = False
        
        self.global_tracks = [
            GlobalTrack(global_id=0),
            GlobalTrack(global_id=1)
        ]


    def models_ready(self):
        '''
        Provjerava jesu li svi Trackovi u oba viewa aktivni i imaju dovoljno ažuriranja
        Color modela da bi se mogli koristiti za odabir trenutnog ownera globalnog identiteta.
        '''
    
        for tracker in self.trackers:
            for track in tracker.tracks:
                if (
                    not track.active 
                    or track.missing_frames != 0 
                    or not track.color_model.is_initialized 
                    or track.color_model.model_updates < self.min_model_updates
                ):
                    return False

        return True
                    
    
    def compute_matching_cost(self):
        '''
        Računa matricu troškova za spajanje trackova iz dva viewa
        na temelju Mahalanobis udaljenosti između njihovih color modela.
        '''
        
        tracker0 = self.trackers[0]
        tracker1 = self.trackers[1]

        cost = np.zeros((2, 2), dtype=np.float32)
        
        for i, track0 in enumerate(tracker0.tracks):
            for j, track1 in enumerate(tracker1.tracks):
                cost[i, j] = (
                    track0.color_model.model_distance(
                        track1.color_model
                    )
                )

        return cost
    
    
    def assign_global_tracks(self):
        '''
        Dodjeljuje globalnim identiteteima Trackove iz dva viewa na 
        temelju cost matrice i rješavanja problema dodjele (assignment 
        problem) pomoću linear_sum_assignment funkcije iz scipy.optimize modula.
        '''
        
        cost = self.compute_matching_cost()
        row_ind, col_ind = linear_sum_assignment(cost)

        for t0, t1 in zip(row_ind, col_ind):
            global_track = self.global_tracks[t0]

            global_track.primary = ViewAssignment(
                view_id=0,
                track=self.trackers[0].tracks[t0]
            )

            global_track.secondary = ViewAssignment(
                view_id=1,
                track=self.trackers[1].tracks[t1]
            )

            self.select_initial_owner(global_track)

        self.initialized = True
        
        
    def select_initial_owner(self, global_track):
        '''
        Odabire početnog vlasnika globalnog identiteta na temelju kvalitete trackova iz oba viewa.
        Vlasnik se odabire tako da se usporedi kvaliteta zadnjeg featurea iz oba tracka. Track 
        s boljom kvalitetom postaje primarni vlasnik globalnog identiteta po uvjetu: 
        Manji std_A + std_B = kvalitetniji feature.  
        '''
        p = global_track.primary
        s = global_track.secondary

        if not p.observed or not s.observed:
            return

        q_primary = p.quality
        q_secondary = s.quality

        if (
            q_primary is None or
            q_secondary is None
        ):
            return

        if q_secondary < q_primary:

            global_track.primary, global_track.secondary = (
                global_track.secondary,
                global_track.primary
            )
            
            
    def swap_owner(self, global_track):
        '''Izvršava izmjenu ownera, tj prmary i secondary view-ova.'''
        
        global_track.primary, global_track.secondary = (
            global_track.secondary,
            global_track.primary
        )

        global_track.owner_age = 0
        global_track.owner_missing = 0
        global_track.switch_counter = 0
        global_track.now_using = "primary"

        global_track.owner_switches += 1
        
    
    def update_owner_state(self, global_track):
        '''
        Ažurira stanje ownera ovisno o kojem je slučaju riječ:
        
        1. Ako je owner izgubio svoj Track na određen broj frame-ova, a secondary
        view je aktivan i ima svoj Track, te je prošlo dovoljno frame-ova da se utvrdi da je 
        owner sigurno izgubljen, izvršava se switch.
        
        2. Owner ne mora izgubiti Track, ali ako se svejedno utvrdilo da je secondary 
        ima kvalitetniji Track u svom view-u tokom određenog broja uzastopnih frameova,  
        isto se izvršava zamjena ownera 
        
        '''
        
        primary = global_track.primary
        secondary = global_track.secondary

        if primary is None or secondary is None:
            return

        if primary.observed:
            global_track.owner_age += 1
            global_track.owner_missing = 0
            global_track.now_using = "primary"

        else:
            global_track.owner_missing += 1
            
            if secondary.observed:
                global_track.now_using = "secondary"
                
                if (
                    global_track.owner_missing >
                    self.owner_timeout
                ):
                    self.swap_owner(global_track)
                    return
                
            else:
                global_track.now_using = "none"
                

        if (
            primary.observed and
            secondary.observed
        ):
            q_primary = primary.quality
            q_secondary = secondary.quality

            d_primary = primary.distance
            d_secondary = secondary.distance

            if (
                q_primary is not None and
                q_secondary is not None and
                d_primary is not None and
                d_secondary is not None and
                q_secondary < q_primary and
                d_secondary < d_primary
            ):
                global_track.switch_counter += 1
                
            else:
                global_track.switch_counter = 0


            if (
                global_track.switch_counter >=
                self.switch_delay
            ):
                self.swap_owner(global_track)
    
    
    def update(self):
        '''
        Glavna funkcija multi-view identity resolvera.

        Prilikom prvog pokretanja čeka dok svi lokalni Trackovi iz oba viewa
        ne budu aktivni i dok njihovi Color modeli nemaju dovoljan broj updateova.
        Nakon toga jednom izvršava globalno povezivanje lokalnih Trackova pomoću
        Hungarian algoritma.

        Nakon inicijalne globalne dodjele u svakom sljedećem frameu procjenjuje
        treba li globalni identitet koristiti primary ili secondary ViewAssignment.
        '''
        
        if not self.initialized:

            if not self.models_ready():
                return

            self.assign_global_tracks()

        for global_track in self.global_tracks:
            self.update_owner_state(global_track)
                   