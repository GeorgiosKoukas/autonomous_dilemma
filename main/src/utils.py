import carla
import random
import math
import pandas as pd


NUM_EPISODES = 3
NUM_GENERATIONS = 50
MIN_PEDS = 1
MAX_PEDS = 4  
MAX_SPEED = 30
MIN_SPEED = 10
MAX_DISTANCE_AV_PED = 30
MIN_DISTANCE_AV_PED = 5
MAX_OFFSET_X = 7  # define maximum x offset
MIN_OFFSET_X = -7  # define minimum x offset
MAX_OFFSET_Y = 12  # define maximum y offset (keeping it near the ego)
MIN_OFFSET_Y = 20 # define minimum y offset (keeping it near the ego)
WEIGHT_COLISSION_SPEED = 0.7
WEIGHT_AGE = 0.3
WEIGHT_OBSTACLE_COLISSION = 0.2
MAGNYFYING_FITNESS = 100

pedestrian_data = pd.read_csv('trolley.csv')

MIN_AGE = pedestrian_data['age'].min()
MAX_AGE = pedestrian_data['age'].max()

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

