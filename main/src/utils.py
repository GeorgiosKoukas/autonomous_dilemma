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
from typing import Dict, Union

NUM_GROUPS = 2
NUM_EPISODES = 1
NUM_MAX_EPISODES = 100

NUM_GENERATIONS = 300
RENDERING = True

NUM_PASSENGERS = 2
ETHICAL_KNOB = 0.8
CAR_SAFETY_FACTOR = 1

MIN_PEDS = 1
MAX_PEDS = 3
MAX_SPEED = 27
MIN_SPEED = 10
MAX_DISTANCE_AV_PED = 30
MIN_DISTANCE_AV_PED = 5

MAX_OFFSET_X = 7  # define maximum x offset
MIN_OFFSET_X = -7  # define minimum x offset
MAX_OFFSET_Y = 17  # define maximum y offset (keeping it near the ego)
MIN_OFFSET_Y = 23  # define minimum y offset (keeping it near the ego)


MAGNYFYING_FITNESS = 100

pedestrian_data = pd.read_csv("trolley.csv")


def score_calculator(results, scenario):
    """
    Calculates the harm score based on the collision results and scenario parameters.

    Args:
        results (dict): A dictionary containing collision results.
        scenario (object): An object representing the scenario.

    Returns:
        tuple: A tuple containing the harm score, normalized pedestrians harm, and normalized passengers harm.
    """
    pedestrians_harm = 0
    passengers_harm = 0

    for pedestrian_hit in results["pedestrian_collisions"]:
        pedestrians_harm += 1 - (
            normalize_input(pedestrian_hit["pedestrian_age"], results["min_age"], results["max_age"])
            ) * normalize_velocity(
            pedestrian_hit["speed"]
        )
        for passenger_age in results["passengers"]["age"]:
            passengers_harm += 1 - (normalize_input(passenger_age, min(results["passengers"]["age"]), max(results["passengers"]["age"]))
         ) * normalize_velocity(pedestrian_hit["speed"])
    for obstacle_hit in results["other_collisions"]:
        for passenger_age in results["passengers"]["age"]:
            passengers_harm += 1 - (normalize_input(passenger_age, min(results["passengers"]["age"]), max(results["passengers"]["age"]))
         ) * normalize_velocity(obstacle_hit["speed"])
    #HAVE TO TAKE INTO ACCOUNT COLISSIONS WITH OBSTACLES OR WALLS
    normalized_pedestrians_harm = pedestrians_harm / scenario.total_pedestrians
    if (len(results["pedestrian_collisions"]) + len(results["other_collisions"])) == 0:
        normalized_passengers_harm = 0
    else:
        normalized_passengers_harm = passengers_harm / len(results["passengers"]["age"]) / (len(results["pedestrian_collisions"]) + len(results["other_collisions"])) / CAR_SAFETY_FACTOR
    # Apply ETHICAL_KNOB
    harm = (ETHICAL_KNOB * normalized_pedestrians_harm + (1 - ETHICAL_KNOB) * normalized_passengers_harm)

    return harm, normalized_pedestrians_harm, normalized_passengers_harm

def save_best_genome(genomes, generation):
    """
    Save the best genome from a generation to a file.

    Parameters:
    genomes (list): List of genomes.
    generation (int): Generation number.

    Returns:
    None
    """
    # Ensure the 'winners' directory exists
    winners_dir = os.path.join(os.path.dirname(__file__), 'winners')
    if not os.path.exists(winners_dir):
        os.makedirs(winners_dir)

    best_genome = max(genomes, key=lambda g: g[1].fitness)
    with open(os.path.join(winners_dir, f"winner_gen_{generation}.pkl"), "wb") as f:
        pickle.dump(best_genome, f, pickle.HIGHEST_PROTOCOL)
        
def normalize_input(value, min_val, max_val):
    """
    Normalize the input value between the minimum and maximum values.

    Parameters:
    value (float): The input value to be normalized.
    min_val (float): The minimum value of the range.
    max_val (float): The maximum value of the range.

    Returns:
    float: The normalized value.
    """
    if max_val == min_val:
        return 0.0   
    return (value - min_val) / (max_val - min_val)
def settings_setter(world):
    """
    Sets the settings for the world.

    Args:
        world: The world object.

    Returns:
        None
    """
    settings = world.get_settings()
    settings.synchronous_mode = True
    weather_params = {
        "cloudiness": 0.0,
        "precipitation": 50.0,
        "sun_altitude_angle": 90.0,
    }
    settings.no_rendering_mode = not RENDERING
    settings.fixed_delta_seconds = 0.05 
    world.apply_settings(settings)
def generate_node_names(max_peds, num_groups):
    """
    Generate node names for pedestrian groups and ego vehicle velocity.

    Args:
        max_peds (int): Maximum number of pedestrians in a group.
        num_groups (int): Number of pedestrian groups.

    Returns:
        dict: A dictionary mapping node keys to node names.
    """
    node_names = {}
    node_counter = 1

    for group_num in range(1, num_groups + 1):
        for ped_num in range(1, max_peds + 1):
            location_key = -(node_counter)
            age_key = -(node_counter + 1)

            location_name = f"group_{group_num}_ped_{ped_num}_location"
            age_name = f"group_{group_num}_ped_{ped_num}_age"

            node_names[location_key] = location_name
            node_names[age_key] = age_name

            node_counter += 2

    # Add ego vehicle velocity
    ego_velocity_key = -(node_counter)
    node_names[ego_velocity_key] = "ego_vehicle_velocity"

    return node_names


def generate_groups_config(NUM_GROUPS):
    """
    Generate a configuration for groups of pedestrians.

    Args:
        NUM_GROUPS (int): The number of groups to generate.

    Returns:
        dict: A dictionary containing the configuration for the groups.
    """
    groups = []
    for _ in range(NUM_GROUPS):
        group_config = {
            "number": MAX_PEDS, #random.randint(MIN_PEDS, MAX_PEDS),
            "rotation": carla.Rotation(pitch=0.462902, yaw=-84.546936, roll=-0.001007),
        }
        groups.append(group_config)

    groups_config = {"groups": groups}
    return groups_config


import math

def calculate_collision_angle(ego_vehicle, other_actor):
    """
    Calculates the collision angle between the ego vehicle and another actor.

    Args:
        ego_vehicle (Vehicle): The ego vehicle.
        other_actor (Actor): The other actor.

    Returns:
        float: The collision angle in degrees.
    """
    ego_velocity = ego_vehicle.get_velocity()
    actor_velocity = other_actor.get_velocity()
    dot_product = (
        ego_velocity.x * actor_velocity.x
        + ego_velocity.y * actor_velocity.y
        + ego_velocity.z * actor_velocity.z
    )
    ego_magnitude = (
        ego_velocity.x**2 + ego_velocity.y**2 + ego_velocity.z**2
    ) ** 0.5
    cos_angle = dot_product / ego_magnitude
    return math.degrees(math.acos(min(max(cos_angle, -1.0), 1.0)))


def set_random_offsets(direction = None):
    """
    Generates random offsets for positioning groups of objects.

    Args:
        direction (str, optional): Direction of the offset. Can be 'right', 'left', or None. Defaults to None.

    Returns:
        tuple: A tuple containing the offset for the first group and the offset for the other groups.
    """
    offset_group_0 = carla.Vector3D(
        0, random.uniform(MIN_OFFSET_Y, MAX_OFFSET_Y), 0
    )  # The first group is always in the middle
    if direction == 'right':
        offset_other_groups = carla.Vector3D(
            random.uniform(MIN_OFFSET_X, 0),
            random.uniform(MIN_OFFSET_Y, MAX_OFFSET_Y),
            0,
        )
    elif direction == 'left':

        offset_other_groups = carla.Vector3D(
            random.uniform(0, MAX_OFFSET_X),
            random.uniform(MIN_OFFSET_Y, MAX_OFFSET_Y),
            0,
        )
    else :
        offset_other_groups = carla.Vector3D(
            random.uniform(MIN_OFFSET_X, MAX_OFFSET_X),
            random.uniform(MIN_OFFSET_Y, MAX_OFFSET_Y),
            0,
        )
    return offset_group_0, offset_other_groups


def normalize_pedestrian_count(count):
    """
    Normalize the pedestrian count to a value between 0 and 1.

    Args:
        count (int): The pedestrian count.

    Returns:
        float: The normalized pedestrian count.
    """
    return (count - MIN_PEDS) / (MAX_PEDS - MIN_PEDS)


def normalize_velocity(velocity):
    """
    Normalize the given velocity by dividing it by the maximum speed.

    Args:
        velocity (float): The velocity to be normalized.

    Returns:
        float: The normalized velocity.
    """
    return velocity / MAX_SPEED


def generate_scenario_attributes(client):
    """
    Generate scenario attributes for a given client.

    Args:
        client: The client object.

    Returns:
        scenario_attributes: A tuple containing the generated scenario attributes.
            - groups_config: The generated groups configuration.
            - client: The client object.
            - pedestrian_data: The sampled pedestrian data.
            - spawn_locations: The generated spawn locations for each group.
            - group_offsets: The generated group offsets.
    """
    groups_config = generate_groups_config(NUM_GROUPS)
    group_offsets = [
        set_random_offsets()[0] if i == 0 else set_random_offsets()[1]
        for i in range(len(groups_config["groups"]) + 1)
    ]
    total_pedestrians = sum([group["number"] for group in groups_config["groups"]])
    scenario_attributes = (
        groups_config,
        client,
        pedestrian_data.sample(total_pedestrians).to_dict("records"),
        [
            [generate_spawn_location() for _ in range(group["number"])]
            for group in groups_config["groups"]
        ],
        group_offsets,
    )
    return scenario_attributes


def normalize_distance(distance):
    """
    Normalize the given distance using the minimum and maximum distance values.
    
    Args:
        distance (float): The distance to be normalized.
    
    Returns:
        float: The normalized distance.
    """
    return (distance - MIN_DISTANCE_AV_PED) / (
        MAX_DISTANCE_AV_PED - MIN_DISTANCE_AV_PED
    )


def clip_value(value, min_val=0, max_val=1):
    """
    Clips the given value between the specified minimum and maximum values.

    Args:
        value (float): The value to be clipped.
        min_val (float, optional): The minimum value allowed. Defaults to 0.
        max_val (float, optional): The maximum value allowed. Defaults to 1.

    Returns:
        float: The clipped value.
    """
    return max(min(value, max_val), min_val)


def generate_spawn_location():
    """
    Generate a random spawn location for a pedestrian.

    Returns:
        carla.Location: A random spawn location for a pedestrian.
    """
    spawn_x = random.uniform(0, 1)
    spawn_y = random.uniform(0, 1)
    spawn_z = random.uniform(0.1, 0.8)
    return carla.Location(spawn_x, spawn_y, spawn_z)


def create_results_dictionary():
    """
    Creates a dictionary to store results related to collisions, pedestrians, steering, passengers, and scenario harm score.

    Returns:
        dict: A dictionary with empty lists for each result category.
    """
    return {
        "collisions": [],  # This will store dictionaries for each collision
        "total_pedestrians": [],
        "pedestrian_ages": [],
        "avg_steering": [],
        "total_passengers": [],
        "passengers_ages": [],
        "scenario_total_harm_score": [],
    }


def pad_steering_values(all_scenario_steering_values):
    """
    Pad the steering values of all scenarios with zeros to make them of equal length.

    Args:
        all_scenario_steering_values (list): A list of lists, where each inner list represents the steering values of a scenario.

    Returns:
        list: A list of lists with padded steering values.
    """
    max_length = max(len(scenario) for scenario in all_scenario_steering_values)
    for scenario in all_scenario_steering_values:
        while len(scenario) < max_length:
            scenario.append(0)  # Appending the last steering value
    return all_scenario_steering_values


def control_live_feed(ego):
    """
    Display the live feed of control values for the ego vehicle.

    Parameters:
    - ego: The ego vehicle object.

    Returns:
    None
    """
    fig, axs = plt.subplots(3, 1)
    while True:
        if ego:
            control = ego.get_control()
            brake = control.brake
            steering = control.steer
            throttle = control.throttle

            axs[0].cla() 
            axs[0].bar(["Throttle"], [throttle])
            axs[0].set_ylim(0, 1)  

            axs[1].cla()
            axs[1].bar(["Brake"], [brake])
            axs[1].set_ylim(0, 1) 

            axs[2].cla()
            axs[2].bar(["Steering"], [steering])
            axs[2].set_ylim(-1, 1)


def plot_average_steering(average_steering):
    """
    Plots the average steering over time for all scenarios.

    Parameters:
    average_steering (list): A list of average steering values.

    Returns:
    None
    """
    plt.figure()
    plt.plot(average_steering, marker="o")
    plt.title("Average Steering Over Time for All Scenarios")
    plt.xlabel("Time Step")
    plt.ylabel("Average Steering Value")
    plt.grid(True)
    plt.show()


def compute_average_steering(all_scenario_steering_values):
    """
    Computes the average steering value over time for a given set of scenarios.

    Args:
        all_scenario_steering_values (list): A list of lists representing the steering values for each scenario.

    Returns:
        list: A list of average steering values over time.
    """
    all_scenario_steering_values = []
    num_scenarios = len(all_scenario_steering_values)
    num_timesteps = len(
        all_scenario_steering_values[0]
    )  # Assuming all scenarios have the same number of timesteps

    average_steering_over_time = []

    for t in range(num_timesteps):
        avg_steer_at_t = (
            sum([scenario[t] for scenario in all_scenario_steering_values])
            / num_scenarios
        )
        average_steering_over_time.append(avg_steer_at_t)

    return average_steering_over_time


def plot_results(results):
    """
    Plots various metrics based on the results of a simulation.

    Args:
        results (dict): A dictionary containing the simulation results.

    Returns:
        None
    """
    # Harm Score over Scenarios
    plt.figure()
    plt.plot(results["harm_scores"], marker="o")
    plt.title("Harm Score Over Scenarios")
    plt.xlabel("Scenario Number")
    plt.ylabel("Harm Score")
    plt.grid(True)

    # Distribution of Harm Scores
    plt.figure()
    plt.hist(results["harm_scores"], bins=20)
    plt.title("Distribution of Harm Scores")
    plt.xlabel("Harm Score")
    plt.ylabel("Frequency")
    plt.grid(True)

    # Pedestrians Hit / Total Pedestrians
    ratios = [
        hit / total
        for hit, total in zip(results["pedestrians_hit"], results["total_pedestrians"])
    ]
    plt.figure()
    plt.plot(ratios, marker="o")
    plt.title("Ratio of Pedestrians Hit Over Scenarios")
    plt.xlabel("Scenario Number")
    plt.ylabel("Pedestrians Hit / Total Pedestrians")
    plt.grid(True)

    # Average Steering Angle Over Scenarios
    plt.figure()
    plt.plot(results["avg_steering"], marker="o")
    plt.title("Average Steering Angle Over Scenarios")
    plt.xlabel("Scenario Number")
    plt.ylabel("Average Steering Angle")
    plt.grid(True)

    # Show all plots
    plt.show()
