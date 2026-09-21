from dataclasses import dataclass
from multiview.view_assignment import ViewAssignment


@dataclass
class GlobalTrack:
    """
    Globalni identitet jednog boksača koji se pojavljuje u više viewa. Identitet se 
    održava kroz ViewAssignment objekte koji povezuju view i pripadajući Track objekt.
    Ne sadrži vlastiti appearance model niti vlastiti tracking. Isključivo održava 
    odluku kojem lokalnom Track objektu trenutno vjerujemo.
    Također, održava statistiku za odabir trenutnog ownera globalnog identiteta.
    """
    
    def __init__(self, global_id):

        self.global_id = global_id
        
        self.now_using = "primary"

        self.primary: ViewAssignment = None
        self.secondary: ViewAssignment = None

        self.switch_counter = 0

        self.owner_age = 0
        self.owner_missing = 0
        self.owner_switches = 0