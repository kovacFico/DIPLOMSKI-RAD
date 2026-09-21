from typing import Dict, List, Tuple
from tracking.track import Track


class IdentityEvaluator:
    """
    Evaluator identiteta za single-view tracking.

    Evaluacija se temelji na ručno označenim intervalima u datoteci ground_truth.py
    u kojima se evidentirajju frameovi gdje pojedini Track prati pogrešnog borca.

    Metrike:
        - Identity Accuracy
        - ID Switches
        - Identity Consistency

    Važno:
        Identity se evaluira samo u frameovima u kojima je track
        stvarno opažen, tj.:

            track.active == True
            AND
            track.missing_frames == 0

        Time se identity accuracy odvaja od coverage metrike.
    """

    def __init__(
        self,
        expected_identity: Dict[int, str],
        wrong_intervals: Dict[int, List[Tuple[int, int]]],
        fps: float
    ):
        self.expected_identity = expected_identity

        self.wrong_intervals = {
            track_id: wrong_intervals.get(track_id, [])
            for track_id in expected_identity
        }

        self.fps = fps

        self.observed_frames = {
            track_id: 0
            for track_id in expected_identity
        }

        self.correct_frames = {
            track_id: 0
            for track_id in expected_identity
        }

        self.wrong_frames = {
            track_id: 0
            for track_id in expected_identity
        }

        # CONSISTENCY
        # Čuvamo duljine kontinuiranih perioda tijekom kojih:
        #   - track je opažen
        #   - identitet je ispravan
        #
        # Ako track nestane ili prijeđe u wrong interval,
        # trenutni stabilni segment završava.

        self.current_stable_segment = {
            track_id: 0
            for track_id in expected_identity
        }

        self.stable_segments = {
            track_id: []
            for track_id in expected_identity
        }

        self.last_frame = None


    def _is_wrong_identity(self, track_id, frame_idx):
        """
        Provjerava nalazi li se frame unutar ručno označenog
        wrong-identity intervala za zadani track.
        """

        for start, end in self.wrong_intervals.get(track_id, []):

            if start <= frame_idx <= end:
                return True

        return False


    def _close_stable_segment(self, track_id):
        """
        Završava trenutni stabilni identity segment.
        """

        length = self.current_stable_segment[track_id]

        if length > 0:
            self.stable_segments[track_id].append(length)

        self.current_stable_segment[track_id] = 0

    def update(
        self,
        frame_idx: int,
        tracks: List[Track]
    ):
        """
        Ažurira evaluator za trenutni frame.
        """

        self.last_frame = frame_idx

        for track in tracks:
            track_id = track.id
            if track_id not in self.expected_identity:
                continue

            observed = (
                track.active
                and track.missing_frames == 0
                and track.bbox is not None
            )

            if not observed:
                self._close_stable_segment(track_id)
                continue

            self.observed_frames[track_id] += 1

            wrong = self._is_wrong_identity(track_id, frame_idx)

            if wrong:
                self.wrong_frames[track_id] += 1
                self._close_stable_segment(track_id)

            else:
                self.correct_frames[track_id] += 1
                self.current_stable_segment[track_id] += 1


    def finalize(self):
        """
        Zatvara stabilne segmente koji su ostali otvoreni
        na kraju videa.
        """

        for track_id in self.expected_identity:
            self._close_stable_segment(track_id)


    def identity_accuracy(self, track_id=None):
        """
        Identity Accuracy:

            correct observed frames
            -----------------------
            all observed frames

        Missing frameovi nisu identity error.
        Njih zasebno mjeri Coverage.
        """

        if track_id is not None:
            observed = self.observed_frames[track_id]
            if observed == 0:
                return 0.0

            return (
                self.correct_frames[track_id]
                / observed
            )

        total_observed = sum(self.observed_frames.values())
        total_correct = sum(self.correct_frames.values())

        if total_observed == 0:
            return 0.0

        return total_correct / total_observed


    def id_switches(self, track_id=None):
        """
        Jedan ground-truth wrong interval predstavlja
        jednu ID-switch epizodu.

        RED -> BLACK = switch

        BLACK -> RED = povratak i NE stvara novi switch.
        """

        if track_id is not None:
            return len(self.wrong_intervals.get(track_id, []))

        return sum(
            len(intervals)
            for intervals in self.wrong_intervals.values()
        )
        

    def identity_consistency_frames(self, track_id=None):
        """
        Prosječna duljina kontinuiranog ispravnog identity
        segmenta izražena u frameovima.
        """

        if track_id is not None:
            segments = self.stable_segments[track_id]

        else:
            segments = []
            for track_segments in self.stable_segments.values():
                segments.extend(track_segments)

        if len(segments) == 0:
            return 0.0

        return sum(segments) / len(segments)

    def identity_consistency_seconds(self, track_id=None):
        """
        Prosječna duljina kontinuiranog ispravnog identity
        segmenta izražena u sekundama.
        """

        frames = self.identity_consistency_frames(track_id)

        if self.fps <= 0:
            return 0.0

        return frames / self.fps


    def report(self):
        """
        Vraća kompletan evaluacijski rezultat.
        """

        result = {
            "overall": {
                "identity_accuracy":
                    self.identity_accuracy(),

                "id_switches":
                    self.id_switches(),

                "identity_consistency_frames":
                    self.identity_consistency_frames(),

                "identity_consistency_seconds":
                    self.identity_consistency_seconds(),
            },

            "tracks": {}
        }

        for track_id in self.expected_identity:
            result["tracks"][track_id] = {
                "identity":
                    self.expected_identity[track_id],

                "observed_frames":
                    self.observed_frames[track_id],

                "correct_frames":
                    self.correct_frames[track_id],

                "wrong_frames":
                    self.wrong_frames[track_id],

                "identity_accuracy":
                    self.identity_accuracy(track_id),

                "id_switches":
                    self.id_switches(track_id),

                "stable_segments":
                    self.stable_segments[track_id],

                "identity_consistency_frames":
                    self.identity_consistency_frames(
                        track_id
                    ),

                "identity_consistency_seconds":
                    self.identity_consistency_seconds(
                        track_id
                    ),
            }

        return result


class CoverageEvaluator:
    """
    Mjeri koliko često je pojedini track stvarno opažen.
    Track se smatra opaženim samo ako:
        active == True
        missing_frames == 0

    Time se izbjegava situacija da track koji nije imao
    detekciju 10 frameova i dalje umjetno povećava coverage
    samo zato što ga Tracker još nije deaktivirao.
    """

    def __init__(self):
        self.total_frames = 0
        self.observed_frames = {}

    def update(self, tracks):
        self.total_frames += 1
        for track in tracks:
            if track.id not in self.observed_frames:
                self.observed_frames[track.id] = 0

            observed = (
                track.active
                and track.missing_frames == 0
                and track.bbox is not None
            )
            if observed:
                self.observed_frames[track.id] += 1

    def report(self):

        result = {}
        for track_id, observed in self.observed_frames.items():
            if self.total_frames == 0:
                coverage = 0.0
            else:
                coverage = (
                    observed /
                    self.total_frames
                )

            result[track_id] = {
                "observed_frames": observed,
                "total_frames": self.total_frames,
                "coverage": coverage
            }

        return result
