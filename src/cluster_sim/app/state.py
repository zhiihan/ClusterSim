from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import jsons


@dataclass
class BrowserState:
    """
    This class controls the default values when the app is loaded.

    This class contains a local state of the web app, representing a persistent state on the user's local browsing section.
    """

    xmax: int = 4
    ymax: int = 4
    zmax: int = 4
    shape: tuple[int, int, int] = (xmax, ymax, zmax)
    p: float = 0.09
    layout: str = "Grid3D"

    seed: Optional[int] = None
    path_clicks: int = 0

    removed_nodes: set[int] = field(default_factory=set)
    log: str = ""
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

    plot_options: Dict[str, bool] = field(
        default_factory=lambda: 
            {'stabilizer' : False, 
            'coord': True, 
            'vop': True,
            'index': True,
            'neighbors': False}
    )

    def to_json(self):
        return jsons.dumps(self)

    @classmethod
    def from_json(cls, json_str):
        return jsons.loads(json_str, cls)
