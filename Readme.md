# Protein Surface Centroid Graphs

Build a centroid-level graph from a `.off` mesh with properties (`<basename>.txt`), then save a NetworkX pickle.

## Requirements
- Python >= 3.10
- `pymesh` **not** installed via pip; use Docker (`pymesh/pymesh`) or your existing micromamba env.
- Install Python deps:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
