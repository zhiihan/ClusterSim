import jsonpickle


class BrowserState:
    """
    This class contains a local state of the web app, representing a persistent state on the user's local browsing section.

    This class controls the default values when the app is loaded.
    """

    def __init__(self) -> None:

        # TODO: This should be a dataclass

        self.xmax = 4
        self.ymax = 4
        self.zmax = 4

        self.shape = [self.xmax, self.ymax, self.zmax]

        self.p = 0.09
        self.seed = None
        self.path_clicks = 0

        self.cubes = None
        self.lattice = None
        self.lattice_edges = None
        self.connected_cubes = None

        self.removed_nodes = [0] * (self.xmax * self.ymax * self.zmax)
        self.log = []  # html version of move_list
        self.move_list = []  # local variable containing moves
        self.camera_state = {
            "scene.camera": {
                "up": {"x": 0, "y": 0, "z": 1},
                "center": {"x": 0, "y": 0, "z": 0},
                "eye": {"x": 1.4, "y": 1.4, "z": 1.3},
                "projection": {"type": "perspective"},
            }
        }

        self.offset = [0, 0, 0]
        self.xoffset, self.yoffset, self.zoffset = self.offset
        self.ncubes = None

    def to_json(self):
        return jsonpickle.encode(self)
