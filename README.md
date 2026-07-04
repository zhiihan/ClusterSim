<!-- SPHINX-START -->

# ClusterSim

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)

**ClusterSim** is an interactive, browser-based simulator for measurement-based quantum computation (MBQC) and stabilizer circuits using a graph state representation.

Live Web Application: [https://clustersim.app](https://clustersim.app)

> Based on the paper *"Fast simulation of stabilizer circuits using a graph state representation"* by Simon Anders and Hans J. Briegel ([arXiv:quant-ph/0504117v2](https://arxiv.org/abs/quant-ph/0504117v2)).

---

## Key Features

- **Interactive 2D Cytoscape Interface (`graph2d_app.py`)**:
  - **Quantum Gate Application**: Apply Pauli gates ($X$, $Y$, $Z$), Clifford gates ($H$, $S$, $S^\dagger$), Controlled operations ($CX$, $CZ$), and Local Complementations ($LC$, $LCR$).
  - **Measurements**: Measure qubits in $X$, $Y$, or $Z$ bases with random or forced outcomes ($0$ or $1$).
  - **Fusion Gates**: Execute 2-qubit fusion gates ($XXZZ$, $XZZX$, $XYYZ$) with success/failure modes and outcome control.
  - **Graph Manipulations**: Add/remove nodes, toggle/delete edges, and copy/duplicate subgraphs.
  - **Compound Group Containers**: Hierarchically group and ungroup selected qubits into parent container boxes.
  - **Snapshot-Based Undo / Redo**: Step backward and forward through any sequence of quantum operations or plot customizations.
  - **Move Log & String Export**: Track step-by-step move history and load/export graph state string representations.
  - **Styling & Export Options**: Customize node labels, preset Cytoscape shapes, snap nodes to a grid, and export to SVG or JSON.

- **3D Grid Visualization (`grid3d_app.py`)**:
  - Interactive 3D lattice cluster state simulation and visualization.

## Installation

### Prerequisites

- Python `>= 3.12`
- [`uv`](https://github.com/astral-sh/uv) (recommended) or standard `pip`

### Install from Source

```bash
git clone https://github.com/zhiihan/ClusterSim.git
cd ClusterSim

# Using pip:
pip install -e .

# Or using uv:
uv sync
```

---

## Quick Start

### 1. Launch the 2D Graph Simulator

```bash
python graph2d_app.py
```

Open your browser and navigate to `http://127.0.0.1:8050/`.

#### Keyboard Shortcuts (2D App)
- **$N$**: Add node
- **$E$**: Toggle edge between selected nodes
- **Shift + $E$**: Delete edge between selected nodes
- **Backspace**: Delete selected node(s)
- **$H$**: Apply Hadamard ($H$) gate
- **$W$**: Apply Local Complementation ($LC$)
- **$G$**: Group / Ungroup selected nodes
- **Ctrl + $C$ / Ctrl + $V$**: Copy and paste/duplicate selected nodes
- **Ctrl + $Z$ / Ctrl + $Y$**: Undo / Redo

### 2. Launch the 3D Grid Simulator

```bash
python grid3d_app.py
```

Open your browser and navigate to `http://127.0.0.1:8050/`.

---

## Project Structure

```text
ClusterSim/
├── assets/                          # Static assets and clientside JS listeners
├── src/
│   └── clustersim/
│       └── simulator/
│           ├── cluster_state.py     # Core ClusterState stabilizer simulation engine
│           └── frontend_cluster_state.py # 2D layout adapter (metadata, groups, undo stack)
│   └── components/
│       ├── components_2d/           # Dash 2D UI components (action panel, Cytoscape, figure)
│       └── components_3d/           # Dash 3D UI components
├── tests/                           # Pytest test suite
├── graph2d_app.py                   # 2D Dash Application Entry Point
├── grid3d_app.py                    # 3D Dash Application Entry Point
├── pyproject.toml                   # Project configuration & dependencies
└── README.md                        # Documentation
```

---

## Testing & Quality

Run the test suite using `pytest`:

```bash
pytest
```

Run code formatting checks with `ruff`:

```bash
ruff format --check
```

---

## References

1. S. Anders, H. J. Briegel, *"Fast simulation of stabilizer circuits using a graph state representation"*, [Phys. Rev. A 73, 022334 (2006)](https://arxiv.org/abs/quant-ph/0504117v2).
2. M. Hein, J. Eisert, H. J. Briegel, *"Multi-party entanglement in graph states"*, [Phys. Rev. A 69, 062311 (2004)](https://arxiv.org/abs/quant-ph/0307130).
3. D. Schlingemann, *"Cluster states, graph states, and stabilizer codes"*, [quant-ph/0111080](https://arxiv.org/abs/quant-ph/0111080).

---

## License

This project is licensed under the [MIT License](LICENSE).
