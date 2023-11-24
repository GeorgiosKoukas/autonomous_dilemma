from utils import *
from trolley_scenario import TrolleyScenario
pedestrian_data = pd.read_csv('trolley.csv')
        


def set_random_offsets():

        offset_group_0 = carla.Vector3D(0, random.uniform(MIN_OFFSET_Y, MAX_OFFSET_Y), 0) # The first group is always in the middle
        offset_other_groups = carla.Vector3D(random.uniform(MIN_OFFSET_X, MAX_OFFSET_X), random.uniform(MIN_OFFSET_Y, MAX_OFFSET_Y), 0)
        return offset_group_0, offset_other_groups
        

  

    
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
            'pedestrian_ages': [],
            'pedestrian_hit_ages': [],
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

def control_live_feed(ego):
    fig, axs = plt.subplots(3, 1) 
    while True:
        if ego:
            control = ego.get_control()
            brake = control.brake
            steering = control.steer
            throttle = control.throttle

            axs[0].cla()  # Clear the previous plot
            axs[0].bar(['Throttle'], [throttle])
            axs[0].set_ylim(0, 1)  # Set y-axis limit for throttle

            axs[1].cla()
            axs[1].bar(['Brake'], [brake])
            axs[1].set_ylim(0, 1)  # Set y-axis limit for brake

            axs[2].cla()
            axs[2].bar(['Steering'], [steering])
            axs[2].set_ylim(-1, 1)  # Set y-axis limit for steering

if __name__ == "__main__":    
    with open('winner_net.pkl', 'rb') as input_file:
        loaded_winner_net = pickle.load(input_file)

    
   
   
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
ages_hit = []
num_scenarios = 100  # or any number of scenarios for every car
for _ in range(num_scenarios):
    groups_config = {
        'groups': [
            {'number': random.randint(MIN_PEDS, MAX_PEDS), 'rotation': carla.Rotation(pitch=0.462902, yaw=-84.546936, roll=-0.001007)},
            {'number': random.randint(MIN_PEDS, MAX_PEDS), 'rotation': carla.Rotation(pitch=0.462902, yaw=-84.546936, roll=-0.001007)},
            {'number': random.randint(MIN_PEDS, MAX_PEDS), 'rotation': carla.Rotation(pitch=0.462902, yaw=-84.546936, roll=-0.001007)},
            #{'number': random.randint(MIN_PEDS, MAX_PEDS), 'rotation': carla.Rotation(pitch=0.462902, yaw=-84.546936, roll=-0.001007)}
            # You can add more groups here by following the same structure
        ]
    }
    group_offsets = [set_random_offsets()[0] if i == 0 else set_random_offsets()[1] for i in range(len(groups_config['groups']) + 1)]
    total_pedestrians = sum([group['number'] for group in groups_config['groups']])
    scenario_attributes = groups_config, client, weather_params, pedestrian_data.sample(total_pedestrians).to_dict('records'), [[generate_spawn_location() for _ in range(group['number'])] for group in groups_config['groups']], group_offsets
    
    # Initialize the scenario with the random attributes
    scenario = TrolleyScenario(*scenario_attributes)


    # Test the loaded_winner_net with this scenario
    scenario.run(loaded_winner_net)
    results['harm_scores'].append(scenario.total_harm_score)
    results['pedestrians_hit'].append(len(scenario.collided_pedestrians))
    results['pedestrian_ages'].append(scenario.pedestrian_ages)
   
    if len(scenario.collided_pedestrians) > 0:
        collided_pedestrians = [x for x in scenario.collided_pedestrians]
        pedestrian_hit_ages = [scenario.pedestrian_attributes[collided_pedestrian]['age'] for collided_pedestrian in collided_pedestrians]
        results['pedestrian_hit_ages'].extend(pedestrian_hit_ages)
        max_age = max(max(results['pedestrian_ages']))
        min_age = min(min(results['pedestrian_ages']))

        try:
            normalized_age_hit = [(age - min_age) / (max_age - min_age) for age in pedestrian_hit_ages]
        
        except ZeroDivisionError:
            normalized_age_hit = [1 for age in pedestrian_hit_ages]

        scenario_age_hit = sum(normalized_age_hit) / len(normalized_age_hit)
        ages_hit.append(scenario_age_hit)

    else:
        ages_hit.append(2)

    results['harm_scores'] = []
    results['pedestrians_hit'] = []
    results['pedestrian_ages'] = []
fig, ax = plt.subplots()

ax.hist(ages_hit, bins=15, linewidth=0.5, edgecolor="white")
plt.show()

    
