# Heuristic Search with Bayesian Neural Networks

This repository contains code to reproduce the experiments from the paper "Utilising Uncertainty for Efficient Learning of Likely-Admissible Heuristics" by Ofir Maron and Benjamin Rosman. The code here is the Python implementation of the original code written in c#.

## Structure

- `data/`: Directory containing the datasets and trained models.
- `heuristics/`: Directory containing heuristic functions.
- `problems/`: Directory containing problem definitions.
- `search_algorithms/`: Directory containing search algorithms.
- `bnn/`: Directory containing Bayesian Neural Network (BNN) code.
- `solver.py`: Main solver logic.
- `main.py`: Entry point to run experiments.
- `README.md`: This file.

## Setup

1. Generate data:
   ```sh
   python generate_data.py

2. Train the BNN model:
   ```sh
   python bnn.py
 
3.  Run experiments:
     ```sh
     python main.py --problem_type puzzle --heuristic_type learned --initial_state 1,2,3,4,5,6,7,8,0 --search_algorithm astar --weight 1.0
