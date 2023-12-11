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


NUM_GROUPS = 3
NUM_EPISODES = 3
NUM_GENERATIONS = 20

NUM_PASSENGERS = 2
ETHICAL_KNOB = 0.9
CAR_SAFETY_FACTOR = 1

MIN_PEDS = 1
MAX_PEDS = 4
MAX_SPEED = 27
MIN_SPEED = 10
MAX_DISTANCE_AV_PED = 30
MIN_DISTANCE_AV_PED = 5

MAX_OFFSET_X = 7  # define maximum x offset
MIN_OFFSET_X = -7  # define minimum x offset
MAX_OFFSET_Y = 12  # define maximum y offset (keeping it near the ego)
MIN_OFFSET_Y = 20  # define minimum y offset (keeping it near the ego)

WEIGHT_COLISSION_SPEED = 0.7
WEIGHT_AGE = 0.3
WEIGHT_OBSTACLE_COLISSION = 0.2

MAGNYFYING_FITNESS = 200

pedestrian_data = pd.read_csv("trolley.csv")

# MIN_AGE = pedestrian_data['age'].min()
# MAX_AGE = pedestrian_data['age'].max()

def score_calculator(results, scenario):
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

    normalized_pedestrians_harm = pedestrians_harm / scenario.total_pedestrians
    normalized_passengers_harm = passengers_harm / len(results["passengers"]["age"]) / CAR_SAFETY_FACTOR

    # Apply ETHICAL_KNOB
    harm = (ETHICAL_KNOB * normalized_pedestrians_harm + (1 - ETHICAL_KNOB) * normalized_passengers_harm)

    return harm
def normalize_input(value, min_val, max_val):
    if max_val == min_val:
        return 0.0   
    return (value - min_val) / (max_val - min_val)
def settings_setter(world):
    settings = world.get_settings()
    settings.synchronous_mode = True
    weather_params = {
        "cloudiness": 0.0,
        "precipitation": 50.0,
        "sun_altitude_angle": 90.0,
    }
    #settings.no_rendering_mode = True
    settings.fixed_delta_seconds = 0.05 
    world.apply_settings(settings)
def generate_node_names(max_peds, num_groups):
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
    groups = []
    for _ in range(NUM_GROUPS):
        group_config = {
            "number": random.randint(MIN_PEDS, MAX_PEDS),
            "rotation": carla.Rotation(pitch=0.462902, yaw=-84.546936, roll=-0.001007),
        }
        groups.append(group_config)

    groups_config = {"groups": groups}
    return groups_config


def calculate_collision_angle(ego_vehicle, other_actor):
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


def set_random_offsets():
    offset_group_0 = carla.Vector3D(
        0, random.uniform(MIN_OFFSET_Y, MAX_OFFSET_Y), 0
    )  # The first group is always in the middle
    offset_other_groups = carla.Vector3D(
        random.uniform(MIN_OFFSET_X, MAX_OFFSET_X),
        random.uniform(MIN_OFFSET_Y, MAX_OFFSET_Y),
        0,
    )
    return offset_group_0, offset_other_groups


def normalize_pedestrian_count(count):
    return (count - MIN_PEDS) / (MAX_PEDS - MIN_PEDS)


def normalize_velocity(velocity):
    return velocity / MAX_SPEED


def generate_scenario_attributes(client):
    scenario_results = create_results_dictionary()
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
    return (distance - MIN_DISTANCE_AV_PED) / (
        MAX_DISTANCE_AV_PED - MIN_DISTANCE_AV_PED
    )


def clip_value(value, min_val=0, max_val=1):
    return max(min(value, max_val), min_val)


def generate_spawn_location():
    spawn_x = random.uniform(0, 1)
    spawn_y = random.uniform(0, 1)
    spawn_z = random.uniform(0.1, 0.8)
    return carla.Location(spawn_x, spawn_y, spawn_z)


def create_results_dictionary():
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
    max_length = max(len(scenario) for scenario in all_scenario_steering_values)
    for scenario in all_scenario_steering_values:
        while len(scenario) < max_length:
            scenario.append(0)  # Appending the last steering value
    return all_scenario_steering_values


def control_live_feed(ego):
    fig, axs = plt.subplots(3, 1)
    while True:
        if ego:
            control = ego.get_control()
            brake = control.brake
            steering = control.steer
            throttle = control.throttle

            axs[0].cla()  # Clear the previous plot
            axs[0].bar(["Throttle"], [throttle])
            axs[0].set_ylim(0, 1)  # Set y-axis limit for throttle

            axs[1].cla()
            axs[1].bar(["Brake"], [brake])
            axs[1].set_ylim(0, 1)  # Set y-axis limit for brake

            axs[2].cla()
            axs[2].bar(["Steering"], [steering])
            axs[2].set_ylim(-1, 1)  # Set y-axis limit for steering


def plot_average_steering(average_steering):
    plt.figure()
    plt.plot(average_steering, marker="o")
    plt.title("Average Steering Over Time for All Scenarios")
    plt.xlabel("Time Step")
    plt.ylabel("Average Steering Value")
    plt.grid(True)
    plt.show()


def compute_average_steering(all_scenario_steering_values):
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
