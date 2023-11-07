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

pedestrian_data = pd.read_csv('trolley.csv')
        
NUM_EPISODES = 10
NUM_GENERATIONS = 100
MIN_PEDS = 1
MAX_PEDS = 4  
MAX_SPEED = 60
MIN_SPEED = 10
MAX_DISTANCE_AV_PED = 30
MIN_DISTANCE_AV_PED = 5
MAX_OFFSET_X = 7  # define maximum x offset
MIN_OFFSET_X = -7  # define minimum x offset
MAX_OFFSET_Y = 15  # define maximum y offset (keeping it near the ego)
MIN_OFFSET_Y = 8 # define minimum y offset (keeping it near the ego)
            
def set_random_offsets():

        offset_group_0 = carla.Vector3D(0, random.uniform(MIN_OFFSET_Y, MAX_OFFSET_Y), 0) # The first group is always in the middle
        offset_other_groups = carla.Vector3D(random.uniform(MIN_OFFSET_X, MAX_OFFSET_X), random.uniform(MIN_OFFSET_Y, MAX_OFFSET_Y), 0)
        return offset_group_0, offset_other_groups
        
def normalize_pedestrian_count(count):
    return (count - MIN_PEDS) / (MAX_PEDS - MIN_PEDS)

def normalize_velocity(velocity):
    return velocity / MAX_SPEED

def normalize_distance(distance):
    return (distance - MIN_DISTANCE_AV_PED) / (MAX_DISTANCE_AV_PED - MIN_DISTANCE_AV_PED)

def clip_value(value, min_val=0, max_val=1):
    return max(min(value, max_val), min_val)

def generate_spawn_location():
    spawn_x = random.uniform(0, 1)
    spawn_y = random.uniform(0, 1)
    spawn_z = random.uniform(0.1, 0.8)
    return carla.Location(spawn_x, spawn_y, spawn_z)
  

    
        
        

      
        
        
        
def plot_results():
    # Harm Score over Scenarios
    plt.figure()
    plt.plot(results['harm_scores'], marker='o')
    plt.title('Harm Score Over Scenarios')
    plt.xlabel('Scenario Number')
    plt.ylabel('Harm Score')
    plt.grid(True)
    
    # Distribution of Harm Scores
    plt.figure()
    plt.hist(results['harm_scores'], bins=20)
    plt.title('Distribution of Harm Scores')
    plt.xlabel('Harm Score')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    # Pedestrians Hit / Total Pedestrians
    ratios = [hit / total for hit, total in zip(results['pedestrians_hit'], results['total_pedestrians'])]
    plt.figure()
    plt.plot(ratios, marker='o')
    plt.title('Ratio of Pedestrians Hit Over Scenarios')
    plt.xlabel('Scenario Number')
    plt.ylabel('Pedestrians Hit / Total Pedestrians')
    plt.grid(True)

    # Average Steering Angle Over Scenarios
    plt.figure()
    plt.plot(results['avg_steering'], marker='o')
    plt.title('Average Steering Angle Over Scenarios')
    plt.xlabel('Scenario Number')
    plt.ylabel('Average Steering Angle')
    plt.grid(True)
    
    # Show all plots
    plt.show()
def compute_average_steering(all_scenario_steering_values):
    num_scenarios = len(all_scenario_steering_values)
    num_timesteps = len(all_scenario_steering_values[0])  # Assuming all scenarios have the same number of timesteps

    average_steering_over_time = []

    for t in range(num_timesteps):
        avg_steer_at_t = sum([scenario[t] for scenario in all_scenario_steering_values]) / num_scenarios
        average_steering_over_time.append(avg_steer_at_t)

    return average_steering_over_time
all_scenario_steering_values = []
results = {
            'harm_scores': [],
            'pedestrians_hit': [],
            'total_pedestrians': [],
            'avg_steering': []
        }  
def plot_average_steering(average_steering):
    plt.figure()
    plt.plot(average_steering, marker='o')
    plt.title('Average Steering Over Time for All Scenarios')
    plt.xlabel('Time Step')
    plt.ylabel('Average Steering Value')
    plt.grid(True)
    plt.show()

def pad_steering_values(all_scenario_steering_values):
    max_length = max(len(scenario) for scenario in all_scenario_steering_values)
    for scenario in all_scenario_steering_values:
        while len(scenario) < max_length:
            scenario.append(0)  # Appending the last steering value
    return all_scenario_steering_values
if __name__ == "__main__":    
    with open('winner_net.pkl', 'rb') as input_file:
        loaded_winner_net = pickle.load(input_file)

    num_scenarios = 30  # or any number of scenarios for every car
   
   
    client = carla.Client('localhost', 2000)
    client.set_timeout(15)
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True
    weather_params = {
        'cloudiness': 0.0,
        'precipitation': 50.0,
        'sun_altitude_angle': 90.0
    }

for _ in range(num_scenarios):
    groups_config = {
        'groups': [
            {'number': random.randint(MIN_PEDS, MAX_PEDS), 'rotation': carla.Rotation(pitch=0.462902, yaw=-84.546936, roll=-0.001007)},
            {'number': random.randint(MIN_PEDS, MAX_PEDS), 'rotation': carla.Rotation(pitch=0.462902, yaw=-84.546936, roll=-0.001007)},
            {'number': random.randint(MIN_PEDS, MAX_PEDS), 'rotation': carla.Rotation(pitch=0.462902, yaw=-84.546936, roll=-0.001007)},
            {'number': random.randint(MIN_PEDS, MAX_PEDS), 'rotation': carla.Rotation(pitch=0.462902, yaw=-84.546936, roll=-0.001007)}
            # You can add more groups here by following the same structure
        ]
    }
    group_offsets = [set_random_offsets()[0] if i == 0 else set_random_offsets()[1] for i in range(len(groups_config['groups']) + 1)]
    total_pedestrians = sum([group['number'] for group in groups_config['groups']])
    max_potential_harm = total_pedestrians * MAX_SPEED  
    scenario_attributes = groups_config, client, weather_params, pedestrian_data.sample(total_pedestrians).to_dict('records'), [[generate_spawn_location() for _ in range(group['number'])] for group in groups_config['groups']], group_offsets
    
    # Initialize the scenario with the random attributes
    scenario = TrolleyScenario(*scenario_attributes)
    
    # Test the loaded_winner_net with this scenario
    scenario.run(loaded_winner_net)
    results['harm_scores'].append(scenario.total_harm_score)
    results['pedestrians_hit'].append(len(scenario.collided_pedestrians))
    
   
#     all_scenario_steering_values.append(scenario.steering_values)
# all_scenario_steering_values = pad_steering_values(all_scenario_steering_values)    
plot_results()
# average_steering = compute_average_steering(all_scenario_steering_values)
# plot_average_steering(average_steering)
    
