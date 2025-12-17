

# HexGridAStar
echo "# HexGridAStar" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/style52hz/HexGridAStar.git
git push -u origin main
 Obstacle Density-Aware A* Path Planning on Hexagonal Grids

This repository provides a reference implementation of the enhanced A* algorithm
with obstacle density awareness for hexagonal grid-based off-road path planning,
as described in the accompanying paper.

## Requirements
- Python 3.8+
- numpy

## Repository Structure
- src/: core algorithm implementation
- examples/: quick test scripts
- data/: demo input data

## Quick Test
To run a simple test case:

```bash
python hexagonal_grid_version1.py
