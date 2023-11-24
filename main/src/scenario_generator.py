#Provoles dianismatwn se reaction pedestrians

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
from utils import *
from configobj import ConfigObj

pedestrian_data = pd.read_csv('trolley.csv')


class TrolleyScenario:

    def __init__(self, groups_config, client, weather, pre_sampled_attributes, generation_spawn_locations, group_offsets):
        
        self.num_groups = len(groups_config['groups'])
        self.setup_variables(groups_config, client)
        self.set_spawn_locations()
        self.weather_params = weather

        self.pedestrian_bp = self.world.get_blueprint_library().filter('*pedestrian*')
        self.obstacle_bp = self.world.get_blueprint_library().filter('*static.prop.vendingmachine*')
        self.vehicle_bp = self.world.get_blueprint_library().filter('*vehicle.tesla.model3*')

        self.spectator = self.world.get_spectator()
        self.pre_sampled_attributes = pre_sampled_attributes
        self.generation_spawn_locations = generation_spawn_locations
        self.spawn_locations = []
        self.actor_id_lists = [[] for _ in range(self.num_groups)]

        self.lanes = [[] for _ in range(self.num_groups)]
        self.group_offsets = group_offsets
        self.obstacle = None
        self.pedestrian_ages = []
        self.reacted_pedestrians = {}

    def setup_variables(self, groups_config, client):
        
        self.groups_config = groups_config
        self.total_pedestrians = sum([group['number'] for group in groups_config['groups']])
        self.actor_list = []
        self.group_actors = [[] for _ in range(self.num_groups)]
        self.client = client 
        self.world = self.client.get_world()
        self.pedestrian_attributes = {}
        self.radius_x, self.radius_y = 0.5, 0.5
        self.collided_pedestrians = set()
        self.total_harm_score = 0

    def set_weather(self):
        
        self.weather = carla.WeatherParameters(**self.weather_params)
        self.world.set_weather(self.weather)


    def set_spawn_locations(self):
        
        self.location_ego = carla.Location(x=-1.860508, y=-185.003555, z=0.5)
        self.rotation_ego = carla.Rotation(pitch=0.0, yaw=90, roll=-0.0)
        self.transform_ego = carla.Transform(self.location_ego, self.rotation_ego)

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

    def spawn_obstacle(self):
        
        ego_transform = self.transform_ego

        forward_vector = ego_transform.get_forward_vector()
        right_vector = ego_transform.get_right_vector()
        up_vector = ego_transform.get_up_vector()

        location_offset = self.group_offsets[-1]

        spawn_x = ego_transform.location.x + random.choice([random.randint(7, 9), -random.randint(7, 9)])
        spawn_y = ego_transform.location.y + random.randint(17, 20)
        spawn_z = ego_transform.location.z + location_offset.z

        spawn_location = carla.Location(spawn_x, spawn_y, spawn_z)
        spawn_rotation = carla.Rotation(pitch=0.0, yaw=-90, roll=-0.0)
        spawn_transform = carla.Transform(spawn_location, spawn_rotation)
        
        self.obstacle = self.world.try_spawn_actor(random.choice(self.obstacle_bp), spawn_transform)

        if self.obstacle:

            self.actor_list.append(self.obstacle)
        else:

            print("Obstacle NOT spawned!!!")

        return spawn_location
    
    def spawn_actors_of_group(self, group_config, group_idx):

        group_list = []
        ego_transform = self.transform_ego

        forward_vector = ego_transform.get_forward_vector()
        right_vector = ego_transform.get_right_vector()
        up_vector = ego_transform.get_up_vector()

        location_offset = self.group_offsets[group_idx]

        spawn_x = ego_transform.location.x + location_offset.x
        spawn_y = ego_transform.location.y + location_offset.y
        spawn_z = ego_transform.location.z + location_offset.z
        spawn_location = carla.Location(spawn_x, spawn_y, spawn_z)

        # Create a transform for the center of the Group
        pedestrian_transform = carla.Transform(spawn_location)
        for idx in range(group_config['number']):
            
            location_offset = self.generation_spawn_locations[group_idx][idx]
            ped_transform = carla.Transform(spawn_location + location_offset, group_config['rotation'])
            actor = self.world.try_spawn_actor(random.choice(self.pedestrian_bp), ped_transform)

            if actor:
                self.actor_list.append(actor)
 
                self.actor_id_lists[group_idx].append(actor.id)
                self.assign_pedestrian_attributes(actor, idx)             
                group_list.append(actor)
                self.pedestrian_ages.append(self.pedestrian_attributes[actor.id]['age'])
                #print(group_list)

        return spawn_location, group_list

    def spawn_actors(self):

        for idx in range(self.num_groups):
            
            spawn_location, group_actors = self.spawn_actors_of_group(self.groups_config['groups'][idx], idx)
            self.spawn_locations.append(spawn_location)
            self.group_actors[idx] = group_actors
            self.spawn_obstacle()

    def spawn_ego(self):

        self.ego = self.world.try_spawn_actor(random.choice(self.vehicle_bp), self.transform_ego)
        transform = self.transform_ego

        if self.ego:
            self.actor_list.append(self.ego)
            return True
        
        else:
            print("Ego vehicle spawn failed")
            return False
            

    def calculate_individual_harm(self, pedestrian_id, collision_data):

        pedestrian = self.pedestrian_attributes[pedestrian_id]
        harm_score = WEIGHT_COLISSION_SPEED * normalize_velocity(collision_data['ego_speed']) + WEIGHT_AGE * normalize_age(collision_data['pedestrian_age'])

        return harm_score
    
    def on_collision(self, event):

        pedestrian_id = event.other_actor.id

        # Check if the actor is in any of the actor_id lists (lanes/groups)
        collided_group = None
        for idx, actor_id_list in enumerate(self.actor_id_lists):
            if pedestrian_id in actor_id_list:
                collided_group = idx
                break

        # If the actor was in any lane/group
        if collided_group is not None:

            if pedestrian_id in self.collided_pedestrians:
                return 
            
            if pedestrian_id == self.obstacle.id:
                self.collided_pedestrians.add(pedestrian_id)
                collision_data = {
                    #'timestamp': event.timestamp,
                    #'location': event.transform.location,
                    'ego_speed': (self.ego.get_velocity().x**2 + self.ego.get_velocity().y**2 + self.ego.get_velocity().z**2)**0.5,
                }
                harm_score = MAGNYFYING_FITNESS * (WEIGHT_COLISSION_SPEED * normalize_velocity(collision_data['ego_speed'])) 
                self.total_harm_score += harm_score
            self.collided_pedestrians.add(pedestrian_id)
            collision_data = {
                #'timestamp': event.timestamp,
                #'location': event.transform.location,
                'ego_speed': (self.ego.get_velocity().x**2 + self.ego.get_velocity().y**2 + self.ego.get_velocity().z**2)**0.5,
                #'collision_angle': calculate_collision_angle(self.ego, event.other_actor)
                'pedestrian_age': self.pedestrian_attributes[pedestrian_id]['age']
            }
                
            harm_score = MAGNYFYING_FITNESS * self.calculate_individual_harm(pedestrian_id, collision_data)
            self.total_harm_score += harm_score
            #print(f"Calculated harm score for pedestrian {pedestrian_id}: {harm_score}")
            

            

    def attach_collision_sensor(self):

        bp = self.world.get_blueprint_library().find('sensor.other.collision')
        transform_relative_to_ego = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.collision_sensor = self.world.spawn_actor(bp, transform_relative_to_ego, attach_to=self.ego)
        self.collision_sensor.listen(lambda event: self.on_collision(event))

    #     rss_sensor_bp = self.world.get_blueprint_library().find('sensor.other.rss')
    #     rss_sensor_transform = carla.Transform(carla.Location(x=1.0, z=1.8))
    #     self.rss_sensor = self.world.spawn_actor(rss_sensor_bp, rss_sensor_transform, attach_to=self.ego)
    #    # self.rss_sensor.set_attribute('debug', 'True')
    #     self.rss_sensor.listen(lambda event: self._on_rss_response(event))
        
    # def _on_rss_response(self, response):

    #     timestamp = response.timestamp
    #     response_valid = response.response_valid
    #     proper_response = response.proper_response
    #     ego_dynamics_on_route = response.ego_dynamics_on_route
    #     rss_state_snapshot = response.rss_state_snapshot
    #     situation_snapshot = response.situation_snapshot
    #     world_model = response.world_model
    #     print(f"Timestamp: {timestamp}, Response valid: {response_valid}, Proper response: {proper_response}, Ego dynamics on route: {ego_dynamics_on_route}, RSS state snapshot: {rss_state_snapshot}, Situation snapshot: {situation_snapshot}, World model: {world_model}")
    def calculate_yaw(self, car_location_x, car_location_y, centroid_x, centroid_y):

        return math.degrees(math.atan2(centroid_y - car_location_y, centroid_x - car_location_x))
    
    def apply_control(self, vehicle, group_decision, steering_decision, braking_decision):
        # Choose the group based on the decision and then compute desired_yaw
        neural_network_steering = 2*steering_decision - 1 # y = 2x-1, [0,1] to [-1,1]
        control = carla.VehicleControl(steer=neural_network_steering, throttle=1.0 - braking_decision, brake=braking_decision)
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
    
    def react_to_approaching_car(self, pedestrian, ego_transform, ego_velocity):
        # Calculate distance between pedestrian and ego vehicle
       
        ped_transform = pedestrian.get_transform()
        distance = ego_transform.location.distance(ped_transform.location)

        # Define a threshold distance to trigger the reaction
        reaction_distance = 10.0  # You can adjust this distance
        ped_velocity = pedestrian.get_velocity()
        if distance < reaction_distance and not self.reacted_pedestrians.get(pedestrian.id, False):
            # Calculate relative velocity between ego vehicle and pedestrian
            
            
            relative_velocity = ((ego_velocity.x - ped_velocity.x )**2 + (ego_velocity.y - ped_velocity.y)**2)**0.5

            # Define a threshold velocity to trigger the reaction
            reaction_velocity_threshold = 5.0  # You can adjust this velocity

            if relative_velocity > reaction_velocity_threshold:

                # Move the pedestrian to the side
                right_vector = ego_transform.get_right_vector()
                scaled_vector = carla.Vector3D(random.uniform(-2, 2)*right_vector.x, random.uniform(-2, 2)*right_vector.y, 0)
                new_location = ped_transform.location + scaled_vector
                
                pedestrian.set_transform(carla.Transform(new_location, pedestrian.get_transform().rotation))
                self.reacted_pedestrians[pedestrian.id] = True

    def get_scenario_results(self):
    
        return self.total_harm_score
    
    def destroy_all(self):

        for actor in self.actor_list:
            actor.destroy()

        self.collision_sensor.destroy()
        #self.rss_sensor.destroy()
        self.actor_list = []
        self.actor_id_lists = [[] for _ in range(self.num_groups)]
        
        #print("All actors destroyed")
        
    
    def run(self, net): 

        #atexit.register(self.destroy_all) for some reason it breaks the script???
        if not self.spawn_ego():
            self.destroy_all()
            return False
        
        self.spawn_actors()
        
        self.terminate_thread = False  # Initialize the flag
        thread = threading.Thread(target=self.move_spectator_with_ego)
        thread.start()
        
        
        self.give_ego_initial_speed(MAX_SPEED)
        self.attach_collision_sensor()
        
        #aggregated_attributes = self.compute_all_aggregated_attributes()
        ticks = 0
        

        while ticks < 200:
            self.world.tick()
            ticks = ticks + 1
            # Get the NEAT decisions

            M = MAX_PEDS
            input_vector = []

            for group in self.group_actors:  # Iterate over each group
                for idx in range(M):
                    if idx < len(group):
                        pedestrian = group[idx]
                        dx, dy = self.calculate_distance(self.ego.get_transform().location, pedestrian.get_transform().location)
                        distance = math.sqrt(dx**2 + dy**2)
                        input_vector.append(distance)
                        
                        input_vector.append(self.pedestrian_attributes[pedestrian.id]['age'])

                        # Call the react_to_approaching_car method for each pedestrian
                        self.react_to_approaching_car(pedestrian, self.ego.get_transform(), self.ego.get_velocity())

                    else:
                        input_vector.append(-9999)
                        input_vector.append(-9999)  # Padding with a large distance value
            dx, dy = self.calculate_distance(self.ego.get_transform().location, self.obstacle.get_transform().location)
            distance = math.sqrt(dx**2 + dy**2)
            input_vector.append(self.get_ego_abs_velocity())
            group_decision, steering_decision, braking_decision = net.activate(input_vector)
           
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
    settings.no_rendering_mode = True
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
    
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    checkpoint = neat.Checkpointer(1, filename_prefix='neat-checkpoint-')
    p.add_reporter(checkpoint)


    winner = p.run(eval_genomes, NUM_GENERATIONS)
    node_names =  generate_node_names(MAX_PEDS, NUM_GROUPS)
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
    config.write() 

    run(config_path)
    














  # def compute_aggregated_attributes(self, lane_list):
    #     aggregated_attributes = {
    #         'average_expected_years_left': 0,
    #         'average_health': 0,
    #         'average_contribution': 0,
    #         'average_dependents': 0
    #     }
        
    #     for actor in lane_list:
    #         attributes = self.pedestrian_attributes[actor.id]
    #         aggregated_attributes['average_expected_years_left'] += attributes['expected_years_left']
    #         aggregated_attributes['average_health'] += 10 if attributes['health'] == 1 else 5
    #         aggregated_attributes['average_contribution'] += 10 if attributes['contribution_to_humanity'] == 1 else 5
    #         aggregated_attributes['average_dependents'] += attributes['no_of_dependant']
        
    #     num_pedestrians = len(lane_list)
    #     for key in aggregated_attributes:
    #         try:
    #             aggregated_attributes[key] /= num_pedestrians
    #         except ZeroDivisionError:
    #             aggregated_attributes[key] /= 1

             
        
    #     return aggregated_attributes
    
    # def compute_all_aggregated_attributes(self):
    #     all_attributes = []
    #     for group in self.group_actors:
    #         all_attributes.append(self.compute_aggregated_attributes(group))
    #     return all_attributes