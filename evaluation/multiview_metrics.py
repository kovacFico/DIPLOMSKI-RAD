class MultiViewIdentityEvaluator:
    '''Evaluator identiteta za multi-view sustav.

    Evaluacija se temelji na ground truth podacima svakog pojedinog viewa.
    Svaki globalni identitet u svakom frameu koristi jedan ViewAssignment,
    odnosno lokalni Track iz jednog od dva viewa.

    Na temelju lokalnog Tracka i ground truth intervala određuje se prati li
    trenutni globalni identitet ispravnog borca.

    Metrike:
        - Identity Accuracy
        - ID Switches
        - Identity Consistency
        - Coverage
        - View Usage

    Identity switch se računa kao ulazak globalnog identiteta u novi interval
    pogrešnog identiteta. Povratak na ispravan identitet ne računa se kao
    dodatni switch.

    Args:
        view_ground_truth (dict): Ground truth podaci za svaki view.
        global_expected_identity (dict): Očekivani RED/BLACK identitet
            svakog globalnog tracka.
        fps (float): Broj frameova u sekundi korišten za pretvaranje
            consistency metrike iz frameova u sekunde.

    State:
        started (bool): Označava je li evaluacija globalnih identiteta počela.
        start_frame (int | None): Prvi frame u kojem je GlobalIdentityResolver
            bio inicijaliziran.
        total_frames (int): Broj evaluiranih multi-view frameova.
        observed_frames (dict): Broj frameova u kojima je globalni identitet
            imao valjan observation.
        correct_frames (dict): Broj frameova s ispravnim identitetom.
        wrong_frames (dict): Broj frameova s pogrešnim identitetom.
        switches (dict): Broj identity switch događaja po globalnom identitetu.
        stable_segments (dict): Duljine kontinuiranih perioda ispravnog identiteta.
        view_usage (dict): Broj frameova u kojima je korišten svaki view.
    '''

    def __init__(
        self,
        view_ground_truth,
        global_expected_identity,
        fps
    ):

        self.view_gt = view_ground_truth
        self.global_expected_identity = global_expected_identity
        self.fps = fps

        self.started = False
        self.start_frame = None
        self.total_frames = 0

        self.observed_frames = {
            gid: 0
            for gid in global_expected_identity
        }

        self.correct_frames = {
            gid: 0
            for gid in global_expected_identity
        }

        self.wrong_frames = {
            gid: 0
            for gid in global_expected_identity
        }

        self.switches = {
            gid: 0
            for gid in global_expected_identity
        }

        self.in_wrong_episode = {
            gid: False
            for gid in global_expected_identity
        }

        self.current_stable_segment = {
            gid: 0
            for gid in global_expected_identity
        }

        self.stable_segments = {
            gid: []
            for gid in global_expected_identity
        }

        self.view_usage = {
            0: 0,
            1: 0
        }
        

    def _is_wrong(
        self,
        view_id,
        track_id,
        frame_idx
    ):
        '''
        Provjerava nalazi li se lokalni Track u ground truth intervalu
        pogrešnog identiteta za zadani frame.

        Args:
            view_id (int): ID viewa.
            track_id (int): ID lokalnog Tracka.
            frame_idx (int): Trenutni frame.

        Returns:
            bool: True ako Track u trenutnom frameu prati pogrešnog borca.
        '''

        intervals = (
            self.view_gt[view_id]
            ["wrong_intervals"]
            .get(track_id, [])
        )

        for start, end in intervals:
            if start <= frame_idx <= end:
                return True

        return False


    @staticmethod
    def _opposite(identity):
        '''
        Vraća suprotni identitet.

        Budući da sustav prati točno dva borca, RED i BLACK,
        pogrešan identitet jednog borca mora odgovarati drugom.

        Args:
            identity (str): "RED" ili "BLACK".

        Returns:
            str: Suprotni identitet.
        '''

        if identity == "RED":
            return "BLACK"

        if identity == "BLACK":
            return "RED"

        raise ValueError(
            f"Unknown identity: {identity}"
        )


    def _actual_local_identity(
        self,
        assignment,
        frame_idx
    ):
        '''
        Određuje stvarni identitet lokalnog Tracka koji trenutno koristi
        globalni identitet.

        Nominalni identitet Tracka određen je njegovim početnim RED/BLACK
        mapiranjem. Ako se trenutni frame nalazi unutar ground truth intervala
        pogrešnog identiteta, vraća se suprotni identitet.

        Args:
            assignment (ViewAssignment): Trenutni ViewAssignment globalnog identiteta.
            frame_idx (int): Trenutni frame.

        Returns:
            str: Stvarni identitet lokalnog Tracka ("RED" ili "BLACK").
        '''

        view_id = assignment.view_id
        track_id = assignment.track.id

        nominal = (
            self.view_gt[view_id]
            ["expected_identity"]
            [track_id]
        )

        if self._is_wrong(
            view_id,
            track_id,
            frame_idx
        ):
            return self._opposite(nominal)

        return nominal


    def _close_segment(self, global_id):
        '''
        Završava trenutni kontinuirani segment ispravnog identiteta
        i sprema njegovu duljinu.

        Args:
            global_id (int): ID globalnog identiteta.
        '''
        
        length = (
            self.current_stable_segment[
                global_id
            ]
        )

        if length > 0:

            self.stable_segments[
                global_id
            ].append(length)

        self.current_stable_segment[
            global_id
        ] = 0


    def update(
        self,
        frame_idx,
        global_identity_resolver,
    ):
        '''
        Ažurira multi-view metrike za trenutni frame.

        Evaluacija počinje tek kada je GlobalIdentityResolver inicijaliziran,
        odnosno kada su uspostavljeni globalni identiteti.

        Za svaki globalni identitet:
            1. Određuje se trenutno korišteni ViewAssignment.
            2. Provjerava se postoji li valjan observation.
            3. Određuje se stvarni RED/BLACK identitet lokalnog Tracka.
            4. Uspoređuje se s očekivanim globalnim identitetom.
            5. Ažuriraju se accuracy, switch, consistency i view usage statistike.

        Args:
            frame_idx (int): Trenutni frame.
            global_identity_resolver (GlobalIdentityResolver):
                Trenutno stanje multi-view sustava.
        '''

        if not global_identity_resolver.initialized:
            return

        if not self.started:
            self.started = True
            self.start_frame = frame_idx

        self.total_frames += 1


        for global_track in global_identity_resolver.global_tracks:
            global_id = global_track.global_id

            if global_track.now_using == "primary":
                assignment = (
                    global_track.primary
                )

            elif global_track.now_using == "secondary":
                assignment = (
                    global_track.secondary
                )

            else:
                assignment = None


            if (
                assignment is None or
                not assignment.observed
            ):
                self._close_segment(global_id)
                continue


            self.observed_frames[global_id] += 1
            self.view_usage[assignment.view_id] += 1
            

            actual_identity = (self._actual_local_identity(
                    assignment,
                    frame_idx)
            )
            expected_identity = (self.global_expected_identity[global_id])

            correct = (
                actual_identity ==
                expected_identity
            )

            if correct:
                self.correct_frames[global_id] += 1
                self.current_stable_segment[global_id] += 1
                self.in_wrong_episode[global_id] = False
                
            else:
                self.wrong_frames[global_id] += 1
                self._close_segment(global_id)

                if not self.in_wrong_episode[global_id]:
                    self.switches[global_id] += 1
                    self.in_wrong_episode[global_id] = True


    def finalize(self):
        '''
        Zatvara sve identity consistency segmente koji su ostali
        otvoreni na kraju videa.
        '''
        
        for global_id in (self.global_expected_identity):
            self._close_segment(global_id)


    def identity_accuracy(self, global_id=None):
        '''
        Računa udio frameova u kojima je globalni identitet bio ispravan.

        Ako je global_id zadan, računa metriku samo za taj globalni Track.
        U suprotnom vraća ukupnu accuracy vrijednost za oba identiteta.

        Args:
            global_id (int | None): Opcionalni globalni ID.

        Returns:
            float: Identity accuracy u rasponu [0, 1].
        '''
        
        if global_id is not None:
            if self.observed_frames[global_id] == 0:
                return 0.0

            return (
                self.correct_frames[global_id]
                /
                self.observed_frames[global_id]
            )


        observed = sum(self.observed_frames.values())
        correct = sum(self.correct_frames.values())

        if observed == 0:
            return 0.0

        return correct / observed


    def coverage(self, global_id=None):
        '''
        Računa udio multi-view frameova u kojima globalni identitet ima
        valjan observation iz barem jednog viewa.

        Args:
            global_id (int | None): Opcionalni globalni ID.

        Returns:
            float: Coverage u rasponu [0, 1].
        '''
        
        if self.total_frames == 0:
            return 0.0

        if global_id is not None:
            return (
                self.observed_frames[global_id]
                /
                self.total_frames
            )

        return (
            sum(self.observed_frames.values())
            /
            (
                self.total_frames *
                len(self.global_expected_identity)
            )
        )


    def consistency_frames(self, global_id=None):
        '''
        Računa prosječnu duljinu kontinuiranih segmenata u kojima
        je globalni identitet bio ispravan.

        Args:
            global_id (int | None): Opcionalni globalni ID.

        Returns:
            float: Prosječna duljina stabilnog segmenta u frameovima.
        '''
        
        if global_id is not None:
            segments = self.stable_segments[global_id]

        else:
            segments = []
            for s in self.stable_segments.values():
                segments.extend(s)

        if not segments:
            return 0.0

        return sum(segments) / len(segments)


    def consistency_seconds(self, global_id=None):
        '''
        Vraća identity consistency izražen u sekundama.

        Args:
            global_id (int | None): Opcionalni globalni ID.

        Returns:
            float: Prosječno trajanje stabilnog identity segmenta u sekundama.
        '''

        if self.fps <= 0:
            return 0.0

        return (
            self.consistency_frames(global_id)
            /
            self.fps
        )


    def report(self):
        '''
        Generira konačni izvještaj multi-view evaluacije.

        Returns:
            dict: Rječnik s ukupnim rezultatima i zasebnim rezultatima
            svakog globalnog identiteta.
        '''
        
        overall = {
            "identity_accuracy":
                self.identity_accuracy(),

            "id_switches":
                sum(self.switches.values()),

            "identity_consistency_frames":
                self.consistency_frames(),

            "identity_consistency_seconds":
                self.consistency_seconds(),

            "coverage":
                self.coverage(),

            "start_frame":
                self.start_frame,

            "view_usage":
                self.view_usage.copy()
        }


        global_tracks = {}

        for gid in self.global_expected_identity:
            global_tracks[gid] = {
                "identity":
                    self.global_expected_identity[
                        gid
                    ],

                "observed_frames":
                    self.observed_frames[gid],

                "correct_frames":
                    self.correct_frames[gid],

                "wrong_frames":
                    self.wrong_frames[gid],

                "identity_accuracy":
                    self.identity_accuracy(gid),

                "id_switches":
                    self.switches[gid],

                "identity_consistency_seconds":
                    self.consistency_seconds(gid),
                    
                "identity_consistency_frames":
                    self.consistency_frames(gid),

                "coverage":
                    self.coverage(gid)
            }


        return {
            "overall": overall,
            "global_tracks": global_tracks
        }