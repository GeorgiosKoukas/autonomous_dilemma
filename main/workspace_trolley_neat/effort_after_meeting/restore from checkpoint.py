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
from experiments import TrolleyScenario
from utils import *

pedestrian_data = pd.read_csv('trolley.csv')


def calculate_collision_angle(ego_vehicle, other_actor):
    ego_velocity = ego_vehicle.get_velocity()
    actor_velocity = other_actor.get_velocity()
    dot_product = ego_velocity.x * actor_velocity.x + ego_velocity.y * actor_velocity.y + ego_velocity.z * actor_velocity.z
    ego_magnitude = (ego_velocity.x**2 + ego_velocity.y**2 + ego_velocity.z**2)**0.5
    cos_angle = dot_product / ego_magnitude
    return math.degrees(math.acos(min(max(cos_angle, -1.0), 1.0)))


def set_random_offsets():

        offset_group_0 = carla.Vector3D(0, random.uniform(MIN_OFFSET_Y, MAX_OFFSET_Y), 0) # The first group is always in the middle
        offset_other_groups = carla.Vector3D(random.uniform(MIN_OFFSET_X, MAX_OFFSET_X), random.uniform(MIN_OFFSET_Y, MAX_OFFSET_Y), 0)
        return offset_group_0, offset_other_groups
        
def normalize_pedestrian_count(count):
    return (count - MIN_PEDS) / (MAX_PEDS - MIN_PEDS)

def normalize_velocity(velocity):
    return velocity / MAX_SPEED

def normalize_age(age):
    return (age - MIN_AGE) / (MAX_AGE - MIN_AGE)

def normalize_distance(distance):
    return (distance - MIN_DISTANCE_AV_PED) / (MAX_DISTANCE_AV_PED - MIN_DISTANCE_AV_PED)

def clip_value(value, min_val=0, max_val=1):
    return max(min(value, max_val), min_val)

def generate_spawn_location():
    spawn_x = random.uniform(0, 1)
    spawn_y = random.uniform(0, 1)
    spawn_z = random.uniform(0.1, 0.8)
    return carla.Location(spawn_x, spawn_y, spawn_z)
  
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
        groups_config = {
        'groups': [
            {'number': random.randint(MIN_PEDS, MAX_PEDS), 'rotation': carla.Rotation(pitch=0.462902, yaw=-84.546936, roll=-0.001007)},
            {'number': random.randint(MIN_PEDS, MAX_PEDS), 'rotation': carla.Rotation(pitch=0.462902, yaw=-84.546936, roll=-0.001007)},
            {'number': random.randint(MIN_PEDS, MAX_PEDS), 'rotation': carla.Rotation(pitch=0.462902, yaw=-84.546936, roll=-0.001007)},
           # {'number': random.randint(MIN_PEDS, MAX_PEDS), 'rotation': carla.Rotation(pitch=0.462902, yaw=-84.546936, roll=-0.001007)}
            # You can add more groups here by following the same structure
        ]
    }
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
    p = checkpoint.restore_checkpoint('neat-checkpoint-80')
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    p.add_reporter(checkpoint)


    winner = p.run(eval_genomes, NUM_GENERATIONS)
    node_names = data_mapping = {
    -1: 'group_1_ped_1_location',
    -2: 'group_1_ped_1_age',
    -3: 'group_1_ped_2_location',
    -4: 'group_1_ped_2_age',
    -5: 'group_1_ped_3_location',
    -6: 'group_1_ped_3_age',
    -7: 'group_1_ped_4_location',
    -8: 'group_1_ped_4_age',
    -9: 'group_2_ped_1_location',
    -10: 'group_2_ped_1_age',
    -11: 'group_2_ped_2_location',
    -12: 'group_2_ped_2_age',
    -13: 'group_2_ped_3_location',
    -14: 'group_2_ped_3_age',
    -15: 'group_2_ped_4_location',
    -16: 'group_2_ped_4_age',
    -17: 'group_3_ped_1_location',
    -18: 'group_3_ped_1_age',
    -19: 'group_3_ped_2_location',
    -20: 'group_3_ped_2_age',
    -21: 'group_3_ped_3_location',
    -22: 'group_3_ped_3_age',
    -23: 'group_3_ped_4_location',
    -24: 'group_3_ped_4_age',
    #-25: 'obstacle_location',
    -25: 'ego_vehicle_velocity',
}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    with open('winner_net.pkl', 'wb') as output:
        pickle.dump(winner_net, output, pickle.HIGHEST_PROTOCOL)
if __name__ == "__main__":    
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    run(config_path)
    
