from dataclasses import dataclass
from typing import Optional

from tracking.track import Track


@dataclass
class ViewAssignment:
    """
    Pomocna klasa za vezu između jednog viewa i pripadajućeg Track objekta.
    Globalni sustav ne održava vlastiti appearance model niti vlastiti tracking, već samo
    održava odluku kojem lokalnom Track objektu trenutno vjerujemo. Ova klasa
    sadrži informacije o tome kojem viewu i Track objektu u tom view-u trenutno vjerujemo.
    """
    def __init__(self, view_id, track):
        self.view_id: int = view_id
        self.track: Track = track

    @property
    def valid(self):

        return (
            self.track is not None and
            self.track.active
        )

    @property
    def observed(self):

        return (
            self.track is not None and
            self.track.active and
            self.track.missing_frames == 0
        )

    @property
    def quality(self):

        if self.track is None:
            return None

        if self.track.last_quality is not None:
            return self.track.last_quality

        feature = self.track.last_feature

        if feature is None:
            return None

        return float(
            feature[2] + feature[3]
        )


    @property
    def distance(self):

        if self.track is None:
            return None

        return self.track.last_distance

    @property
    def feature(self):

        if self.track is None:
            return None

        return self.track.last_feature

    @property
    def missing(self):

        if self.track is None:
            return 999

        return self.track.missing_frames