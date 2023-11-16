import carla
import time
import atexit
import random
import math
import pandas as pd
import neat
import os
import threading
import visualize
import matplotlib.pyplot as plt
import numpy as np
import pickle
from configobj import ConfigObj
from scenario_generator import TrolleyScenario
from utils import *

pedestrian_data = pd.read_csv('trolley.csv')


  
def eval_genomes(genomes, config):
    client = carla.Client('localhost', 2000)
    client.set_timeout(15)
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.no_rendering_mode = True
    weather_params = {
        'cloudiness': 0.0,
        'precipitation': 50.0,
        'sun_altitude_angle': 90.0
    }
    #settings.no_rendering_mode = True
    world.apply_settings(settings)
    

    generation_scenarios = []
    for scenario in range(NUM_EPISODES):
        groups_config = generate_groups_config(NUM_GROUPS)
        group_offsets = [set_random_offsets()[0] if i == 0 else set_random_offsets()[1] for i in range(len(groups_config['groups']) + 1)]
        total_pedestrians = sum([group['number'] for group in groups_config['groups']])
        generation_scenarios.append((groups_config, client, weather_params, pedestrian_data.sample(total_pedestrians).to_dict('records'), [[generate_spawn_location() for _ in range(group['number'])] for group in groups_config['groups']], group_offsets))
   
    
    for genome_id, genome in genomes:
        
        genome_fitness = []
        genome.fitness = 0
        for attributes in range(NUM_EPISODES):
            scenario_attributes = generation_scenarios[attributes]
          
            net = neat.nn.FeedForwardNetwork.create(genome, config)   
            # Generate the same scenario for each AV in the same generation
            scenario = TrolleyScenario(*scenario_attributes)
            scenario.run(net)   
            # Use the results to determine the loss
            harm_score = scenario.get_scenario_results()
            genome_fitness.append(-harm_score)
        genome.fitness = sum(genome_fitness)
        print(f"Genome {genome_id} fitness: {genome.fitness}")
    
        
        

        
        
        
        
def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    
    checkpoint = neat.Checkpointer(1, filename_prefix='neat-checkpoint-')
    p = checkpoint.restore_checkpoint('neat-checkpoint-71')
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    p.add_reporter(checkpoint)


    winner = p.run(eval_genomes, NUM_GENERATIONS)
    node_names = generate_node_names(MAX_PEDS, NUM_GROUPS)
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    with open('winner_net.pkl', 'wb') as output:
        pickle.dump(winner_net, output, pickle.HIGHEST_PROTOCOL)
        
if __name__ == "__main__":  
    
    
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
     # Update the value of num_inputs
    config = ConfigObj(config_path, write_empty_values=True)
    num_inputs = NUM_GROUPS * MAX_PEDS * 2 + 1 
    config['DefaultGenome']['num_inputs'] = num_inputs

    # Write the updated configuration back to the file
    config.write() 

    run(config_path)
    
