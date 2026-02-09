import jsonpickle
from dataclasses import dataclass, field
from typing import Any, List, Dict


@dataclass
class BrowserState:
    """
    This class controls the default values when the app is loaded.

    This class contains a local state of the web app, representing a persistent state on the user's local browsing section.
    """

    xmax: int = 4
    ymax: int = 4
    zmax: int = 4
    shape: tuple = (xmax, ymax, zmax)
    p: float = 0.09

    seed: None | int = None
    path_clicks: int = 0

    lattice: dict[str, Any] = None
    lattice_edges: dict[str, Any] = None
    connected_cubes: dict[str, Any] = None

    removed_nodes: List[int] = field(init=False)
    log: List[Any] = field(default_factory=list)
    move_list: List[Any] = field(default_factory=list)
    camera_state: Dict[str, Any] = field(
        default_factory=lambda: {
            "scene.camera": {
                "up": {"x": 0, "y": 0, "z": 1},
                "center": {"x": 0, "y": 0, "z": 0},
                "eye": {"x": 1.4, "y": 1.4, "z": 1.3},
                "projection": {"type": "perspective"},
            }
        }
    )

    offset: tuple = (0, 0, 0)
    xoffset, yoffset, zoffset = offset

    def __post_init__(self) -> None:
        self.removed_nodes = [0] * (self.xmax * self.ymax * self.zmax)

    def to_json(self):
        return jsonpickle.encode(self)
