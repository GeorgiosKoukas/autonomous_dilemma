# TrolleyScenario NEAT-based Autonomous Vehicle Simulation

This repository contains a Python script for producing and simulating autonomous vehicle scenarios using the NEAT (NeuroEvolution of Augmenting Topologies) algorithm. The script utilizes the CARLA simulator for creating realistic scenarios involving pedestrians and obstacles. 

## Getting Started

### Prerequisites
- [CARLA Simulator](https://carla.org/) installed and running on your machine.
- Python 3.6.X installed.
- Required Python packages installed. You can install them using the following command:

    ```bash
    pip install carla pandas neat-python matplotlib scipy networkx


### Usage

1. Clone the repository to your local machine:
    ```bash
    git clone https://github.com/GeorgiosKoukas/autonomous_dilemma


2. Navigate to the project directory

3. Run the CARLA Simulator
   
3. Put the corresponding number port (usually "2000") and run the main.py script to produce neural networks 

4. You can test the generated networks with the test_winner.py script


## Overview

The `TrolleyScenario` class in the script represents a scenario where an autonomous vehicle (AV) navigates through a simulated environment populated with pedestrians and an obstacle. The AV's decision-making is based on the NEAT algorithm, evolving neural networks to control the AV.

### Key Components

- **NEAT Algorithm Integration**: The script uses the NEAT algorithm to evolve neural networks that control the AV's decision-making.

- **CARLA Simulation**: The CARLA simulator is used to create a realistic environment with pedestrians, obstacles, and the AV.

- **Pedestrian and Obstacle Spawning**: Pedestrians and obstacles are spawned in the environment based on configurable parameters.

- **Collision Handling**: The script handles collisions between the AV and pedestrians/obstacles, calculating harm scores based on collision speed and pedestrian age.

- **Fitness Evaluation**: The fitness of each evolved neural network is evaluated based on the harm scores accumulated over multiple simulation episodes.

### Configuration

- The simulation parameters, such as the number of generations, episodes per generation, and the configuration of NEAT, can be adjusted in the `config.txt` file.

- The CARLA simulator settings, such as weather conditions and rendering options, are configurable within the script.


## Acknowledgments

- This script is built on the [CARLA Python API](https://carla.readthedocs.io/).
- The NEAT algorithm implementation is based on the [neat-python](https://neat-python.readthedocs.io/) library.

Feel free to explore and modify the script for your own experiments and simulations. If you encounter any issues or have suggestions for improvement, please open an [issue](https://github.com/GeorgiosKoukas/autonomous_dilemma/issues). Happy simulating!


