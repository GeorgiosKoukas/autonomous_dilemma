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


NUM_GENERATIONS = 20
MIN_PEDS = 1
MAX_PEDS = 5
MAX_SPEED = 60
MIN_SPEED = 10
MAX_DISTANCE_AV_PED = 30
MIN_DISTANCE_AV_PED = 5

pedestrian_data = pd.read_csv('trolley.csv')

def calculate_collision_angle(ego_vehicle, other_actor):
    ego_velocity = ego_vehicle.get_velocity()
    actor_velocity = other_actor.get_velocity()
    dot_product = ego_velocity.x * actor_velocity.x + ego_velocity.y * actor_velocity.y + ego_velocity.z * actor_velocity.z
    ego_magnitude = (ego_velocity.x**2 + ego_velocity.y**2 + ego_velocity.z**2)**0.5
    cos_angle = dot_product / ego_magnitude
    return math.degrees(math.acos(min(max(cos_angle, -1.0), 1.0)))

class TrolleyScenario:
    def __init__(self, number_lane_1, number_lane_2, client, weather, pre_sampled_attributes, generation_spawn_locations):
        self.setup_variables(number_lane_1, number_lane_2, client)
        self.weather_params = weather
        self.pedestrian_bp = self.world.get_blueprint_library().filter('*pedestrian*')
        self.vehicle_bp = self.world.get_blueprint_library().filter('*vehicle.tesla.model3*')
        self.spectator = self.world.get_spectator()
        self.collided_pedestrians = set()
        self.total_harm_score = 0
        self.offset_1 = carla.Vector3D(4, 17, 0) #y is the distance facing the ego, x is the distance perpendicular to the ego
        self.offset_2 = carla.Vector3D(-4, 17, 0)
        self.pre_sampled_attributes = pre_sampled_attributes
        self.generation_spawn_locations = generation_spawn_locations
        self.spawn_location_1, self.spawn_location_2 = None, None
    def setup_variables(self, number_lane_1, number_lane_2, client):
        self.number_lane_1, self.number_lane_2 = number_lane_1, number_lane_2
        self.actor_list, self.lane_1_list, self.lane_2_list = [], [], []
        self.actor_id_list_lane_1, self.actor_id_list_lane_2 = [], []
        self.client = client 
        self.world = self.client.get_world()
        self.pedestrian_attributes = {}
        self.radius_x, self.radius_y = 0.5, 0.5
        self.set_spawn_locations()

    def set_weather(self):
        self.weather = carla.WeatherParameters(**self.weather_params)
        self.world.set_weather(self.weather)

    def set_spawn_locations(self):
        
        self.location_ego = carla.Location(x=-1.860508, y=-185.003555, z=0.5)
        self.rotation_ego = carla.Rotation(pitch=0.0, yaw=90, roll=-0.0)
        self.transform_ego = carla.Transform(self.location_ego, self.rotation_ego)
        self.rotation_lane_1 = carla.Rotation(pitch=0.462902, yaw=-84.546936, roll=-0.001007)
        self.rotation_lane_2 = carla.Rotation(pitch=-7.172853, yaw=-89.619049, roll=-0.001006)

    def teleport_spectator(self, location):
        self.spectator.set_transform(location)
        
    def move_spectator_with_ego(self):
        while not self.terminate_thread:
            time.sleep(0.02)  # A slight delay to avoid excessive updates

            ego_transform = self.ego.get_transform()

            # Calculating the backward vector relative to ego
            forward_vector = ego_transform.get_forward_vector()
            backward_vector = carla.Vector3D(-1 * forward_vector.x, -1 * forward_vector.y, 0)  # Reverse direction

            # Calculating the relative position of the spectator
            relative_location = carla.Location(
                x=ego_transform.location.x + backward_vector.x * 10,  # 10 meters behind
                y=ego_transform.location.y + backward_vector.y * 10, 
                z=ego_transform.location.z + 5  # 5 meters above
            )

            # Set the spectator's transform
            spectator_transform = carla.Transform(
                relative_location,
                carla.Rotation(pitch=-15, yaw=ego_transform.rotation.yaw, roll=0)
            )
            self.spectator.set_transform(spectator_transform)
                
                

    def assign_pedestrian_attributes(self, actor, index):
        self.pedestrian_attributes[actor.id] = self.pre_sampled_attributes[index]

    def spawn_pedestrians_in_lane(self, number, rotation, offset):
        lane_list = []
        ego_transform = self.transform_ego
        forward_vector = ego_transform.get_forward_vector()
        right_vector = ego_transform.get_right_vector()
        up_vector = ego_transform.get_up_vector()
        
        # Compute the new spawn location based on the offset
        spawn_x = ego_transform.location.x + offset.x
        spawn_y = ego_transform.location.y + offset.y
        spawn_z = ego_transform.location.z + offset.z
        spawn_location = carla.Location(spawn_x, spawn_y, spawn_z)
        # Create a transform for the pedestrian
        pedestrian_transform = carla.Transform(spawn_location)
        for idx in range(number):
            lateral_offset = random.uniform(-self.radius_y, self.radius_y)
            longitudinal_offset = random.uniform(-self.radius_y, self.radius_y)
            height_offset = random.uniform(0.1, 0.6)
            spawn_location = spawn_location + self.generation_spawn_locations[idx]
            ped_transform = carla.Transform(spawn_location, rotation)
            actor = self.world.try_spawn_actor(random.choice(self.pedestrian_bp), ped_transform)
            if actor:
                self.actor_list.append(actor)
                lane_list.append(actor)
                self.assign_pedestrian_attributes(actor, idx)
                self.lane_1_list.append(actor) if number == self.number_lane_1 else self.lane_2_list.append(actor)
                self.actor_id_list_lane_1.append(actor.id) if number == self.number_lane_1 else self.actor_id_list_lane_2.append(actor.id)
        

        return spawn_location

    def spawn_pedestrians(self):

        self.spawn_location_1 = self.spawn_pedestrians_in_lane(self.number_lane_1, self.rotation_lane_1, self.offset_1)
        self.spawn_location_2 = self.spawn_pedestrians_in_lane(self.number_lane_2, self.rotation_lane_2, self.offset_2)
    
    def spawn_ego(self):
        self.ego = self.world.try_spawn_actor(random.choice(self.vehicle_bp), self.transform_ego)
        transform = self.transform_ego
        if self.ego:
            self.actor_list.append(self.ego)
        else:
            print("Ego vehicle spawn failed")
        return transform
            

    def calculate_individual_harm(self, pedestrian_id, collision_data):
        pedestrian = self.pedestrian_attributes[pedestrian_id]
        harm_score = sum([
            pedestrian['expected_years_left'],
            10 if pedestrian['health'] == 1 else 5,
            10 if pedestrian['contribution_to_humanity'] == 1 else 5,
            pedestrian['no_of_dependant'],
            collision_data['ego_speed'] * 0.5,
            20 if 0 <= collision_data['collision_angle'] <= 30 else 0,
            collision_data['ego_speed']]) 
        #harm_score = collision_data['ego_speed'] + collision_data['collision_angle']
        
        
        return harm_score
    
    def on_collision(self, event):
        pedestrian_id = event.other_actor.id
        if event.other_actor.id in self.actor_id_list_lane_1 or event.other_actor.id in self.actor_id_list_lane_2:
            if pedestrian_id in self.collided_pedestrians:
                return  
            self.collided_pedestrians.add(pedestrian_id)
            collision_data = {
                'timestamp': event.timestamp,
                'location': event.transform.location,
                'ego_speed': (self.ego.get_velocity().x**2 + self.ego.get_velocity().y**2 + self.ego.get_velocity().z**2)**0.5,
                'collision_angle': calculate_collision_angle(self.ego, event.other_actor)

            }
            
            harm_score = self.calculate_individual_harm(pedestrian_id, collision_data)
            self.total_harm_score += harm_score
            print(f"Calculated harm score for pedestrian {pedestrian_id}: {harm_score}")
            

    def attach_collision_sensor(self):
        bp = self.world.get_blueprint_library().find('sensor.other.collision')
        transform_relative_to_ego = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.collision_sensor = self.world.spawn_actor(bp, transform_relative_to_ego, attach_to=self.ego)
        self.collision_sensor.listen(lambda event: self.on_collision(event))

    def calculate_yaw(self, car_location_x, car_location_y, centroid_x, centroid_y):
        return math.degrees(math.atan2(centroid_y - car_location_y, centroid_x - car_location_x))
    
    def apply_control(self, vehicle, group_decision, steering_decision, braking_decision):
                current_yaw = vehicle.get_transform().rotation.yaw
                desired_yaw = self.calculate_yaw(vehicle.get_transform().location.x, vehicle.get_transform().location.y, 
                                                self.spawn_location_1.x if group_decision < 0.5 else self.spawn_location_2.x, 
                                                self.spawn_location_2.y if group_decision < 0.5 else self.spawn_location_2.y)
                Kp = 0.01
                yaw_error = (desired_yaw - current_yaw + 180) % 360 - 180
                yaw_error_based_steering = Kp * yaw_error
                alpha = 0.1
                neural_network_steering = 2*steering_decision - 1 # y = 2x-1, [0,1] to [-1,1]
                final_steer_command = alpha * yaw_error_based_steering + (1 - alpha) * neural_network_steering #final_steer_command=α×yaw_error_based_steering+(1−α)×neural_network_steering

                #print(f"Group Decision: {group_decision}, Steering Decision: {steering_decision}, Braking Decision: {braking_decision}")
    
                #final_steer_command = neural_network_steering
                control = carla.VehicleControl(steer=final_steer_command, throttle=1.0 - braking_decision, brake=braking_decision)
                vehicle.apply_control(control)
    def give_ego_initial_speed(self, speed):
        self.ego.set_target_velocity(carla.Vector3D(0, speed, 0))
        
    def get_ego_abs_velocity(self):
        velocity = self.ego.get_velocity()
        magnitude = (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
        return magnitude
    def calculate_distance(self, location1, location2):
    
        dx = location1.x - location2.x
        dy = location1.y - location2.y
        
        return dx, dy  
    def get_scenario_results(self):
    # Return the total harm score and other relevant metrics
        return self.total_harm_score
    
    def destroy_all(self):
        for actor in self.actor_list:
            actor.destroy()
        self.lane_1_list, self.lane_2_list = [], []
        self.actor_id_list_lane_1, self.actor_id_list_lane_2 = [], []
        print("All actors destroyed")
        
    def compute_aggregated_attributes(self, lane_list):
        aggregated_attributes = {
            'average_expected_years_left': 0,
            'average_health': 0,
            'average_contribution': 0,
            'average_dependents': 0
        }
        
        for actor in lane_list:
            attributes = self.pedestrian_attributes[actor.id]
            aggregated_attributes['average_expected_years_left'] += attributes['expected_years_left']
            aggregated_attributes['average_health'] += 10 if attributes['health'] == 1 else 5
            aggregated_attributes['average_contribution'] += 10 if attributes['contribution_to_humanity'] == 1 else 5
            aggregated_attributes['average_dependents'] += attributes['no_of_dependant']
        
        num_pedestrians = len(lane_list)
        for key in aggregated_attributes:
            try:
                aggregated_attributes[key] /= num_pedestrians
            except ZeroDivisionError:
                aggregated_attributes[key] /= 1

             
        
        return aggregated_attributes
    def run(self, net): 
        #atexit.register(self.destroy_all) for some reason it breaks the script???
        self.spawn_ego()
        self.spawn_pedestrians()
        
        
        self.terminate_thread = False  # Initialize the flag
        thread = threading.Thread(target=self.move_spectator_with_ego)
        thread.start()
        
        
        self.give_ego_initial_speed(27)
        self.attach_collision_sensor()
        
        self.lane_1_attributes = self.compute_aggregated_attributes(self.lane_1_list)
        self.lane_2_attributes = self.compute_aggregated_attributes(self.lane_2_list)
        ticks = 0
        while ticks < 200:
            self.world.tick()
            ticks = ticks + 1
            # Get the NEAT decisions
            dx1, dy1 = self.calculate_distance(self.ego.get_transform().location, self.spawn_location_1)
            dx2, dy2 = self.calculate_distance(self.ego.get_transform().location, self.spawn_location_2)
            group_decision, steering_decision, braking_decision = net.activate([
                self.number_lane_1,
                self.number_lane_2,
                self.get_ego_abs_velocity(),
                dx1, 
                dy1, 
                dx2, 
                dy2,
                # clip_value(self.lane_1_attributes['average_expected_years_left']),
                # clip_value(self.lane_1_attributes['average_health']),
                # clip_value(self.lane_1_attributes['average_contribution']),
                # clip_value(self.lane_1_attributes['average_dependents']),
                # clip_value(self.lane_2_attributes['average_expected_years_left']),
                # clip_value(self.lane_2_attributes['average_health']),
                # clip_value(self.lane_2_attributes['average_contribution']),
                # clip_value(self.lane_2_attributes['average_dependents'])
            ])
        
            if(len(self.collided_pedestrians) < 1):
                self.apply_control(self.ego, group_decision, steering_decision, braking_decision)
            else:
                control = carla.VehicleControl(steer=0, throttle=0, brake=1)
                self.ego.apply_control(control)
            if(self.get_ego_abs_velocity() < 0.1):
                break
        self.terminate_thread = True
        thread.join()
        self.destroy_all()
        

            

        
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
  
def eval_genomes(genomes, config):
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
    world.apply_settings(settings)
    num_pedestrians = MAX_PEDS + MAX_PEDS  # Maximum number of pedestrians (both lanes)
    
    
    
    
    num_lane_1, num_lane_2 = random.randint(MIN_PEDS, MAX_PEDS), random.randint(MIN_PEDS, MAX_PEDS)
    generation_spawn_locations = [generate_spawn_location() for _ in range(num_lane_1 + num_lane_2)]
    generation_pedestrian_attributes = pedestrian_data.sample(num_lane_1 + num_lane_2).to_dict('records')
    scenario_attributes = num_lane_1, num_lane_2, client, weather_params, generation_pedestrian_attributes, generation_spawn_locations
    for genome_id, genome in genomes:
        reward = 0
        genome.fitness = 0  # start with fitness level of 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        max_potential_harm = (num_lane_1 + num_lane_2) * MAX_SPEED  
        # Generate the same scenario for each AV in the same generation
        scenario = TrolleyScenario(*scenario_attributes)
        scenario.run(net)   
        # Use the results to determine the loss
        harm_score = scenario.get_scenario_results()
        if scenario.collided_pedestrians:  # assuming this checks if no pedestrian was hit
            reward = 100  # add a penalty for not hitting any pedestrians
        normalized_harm = harm_score / max_potential_harm if max_potential_harm != 0 else harm_score

        genome.fitness = reward - harm_score
        print(f"Genome {genome_id} fitness: {genome.fitness}")
    
        
        

        
        
        
        
def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(eval_genomes, NUM_GENERATIONS)
    node_names = {-1:'num_lane_1', -2: 'num_lane_2', -3: 'ego_velocity', -4: 'dx1', -5: 'dy1', -6: 'dx2', -7: 'dy2'}
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
    
