from utils import *
from trolley_scenario import TrolleyScenario
pedestrian_data = pd.read_csv('trolley.csv')
        
results_neat_model, results_left_model, results_right_model, results_straight_model = (create_results_dictionary() for _ in range(4))

 if __name__ == "__main__":    
    with open('winner_net.pkl', 'rb') as input_file:
        loaded_winner_net = pickle.load(input_file)

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
    world.apply_settings(settings)
    ages_hit = []
    num_scenarios = 300  # or any number of scenarios for every car
    for _ in range(num_scenarios):
        groups_config = generate_groups_config(NUM_GROUPS)
        group_offsets = [set_random_offsets()[0] if i == 0 else set_random_offsets()[1] for i in range(len(groups_config['groups']) + 1)]
        total_pedestrians = sum([group['number'] for group in groups_config['groups']])
        scenario_attributes = groups_config, client, weather_params, pedestrian_data.sample(total_pedestrians).to_dict('records'), [[generate_spawn_location() for _ in range(group['number'])] for group in groups_config['groups']], group_offsets
        
        # Initialize the scenario with the random attributes
        scenario = TrolleyScenario(*scenario_attributes)


        # Test the loaded_winner_net with this scenario
        scenario.run(loaded_winner_net)

        results_neat_model['harm_scores'].append(scenario.total_harm_score)
        results_neat_model['pedestrians_hit'].append(len(scenario.collided_pedestrians))
        results_neat_model['pedestrian_ages'].append(scenario.pedestrian_ages)
    
        if len(scenario.collided_pedestrians) > 0:
            collided_pedestrians = [x for x in scenario.collided_pedestrians]
            pedestrian_hit_ages = [scenario.pedestrian_attributes[collided_pedestrian]['age'] for collided_pedestrian in collided_pedestrians]
            results_neat_model['pedestrian_hit_ages'].extend(pedestrian_hit_ages)
            max_age = max(max(results_neat_model['pedestrian_ages']))
            min_age = min(min(results_neat_model['pedestrian_ages']))

            try:
                normalized_age_hit = [(age - min_age) / (max_age - min_age) for age in pedestrian_hit_ages]
            
            except ZeroDivisionError:
                normalized_age_hit = [1 for age in pedestrian_hit_ages]

            scenario_age_hit = sum(normalized_age_hit) / len(normalized_age_hit)
            ages_hit.append(scenario_age_hit)

        else:
            ages_hit.append(2)

        results_neat_model['harm_scores'] = []
        results_neat_model['pedestrians_hit'] = []
        results_neat_model['pedestrian_ages'] = []

        scenario.trivial_run('left')

        results_left_model['harm_scores'].append(scenario.total_harm_score)
        results_left_model['pedestrians_hit'].append(len(scenario.collided_pedestrians))
        results_left_model['pedestrian_ages'].append(scenario.pedestrian_ages)
    
        if len(scenario.collided_pedestrians) > 0:
            collided_pedestrians = [x for x in scenario.collided_pedestrians]
            pedestrian_hit_ages = [scenario.pedestrian_attributes[collided_pedestrian]['age'] for collided_pedestrian in collided_pedestrians]
            results_left_model['pedestrian_hit_ages'].extend(pedestrian_hit_ages)
            max_age = max(max(results_left_model['pedestrian_ages']))
            min_age = min(min(results_left_model['pedestrian_ages']))

            try:
                normalized_age_hit = [(age - min_age) / (max_age - min_age) for age in pedestrian_hit_ages]
            
            except ZeroDivisionError:
                normalized_age_hit = [1 for age in pedestrian_hit_ages]

            scenario_age_hit = sum(normalized_age_hit) / len(normalized_age_hit)
            ages_hit.append(scenario_age_hit)

        else:
            ages_hit.append(2)

        results_left_model['harm_scores'] = []
        results_left_model['pedestrians_hit'] = []
        results_left_model['pedestrian_ages'] = []


        scenario.trivial_run('right')

        results_right_model['harm_scores'].append(scenario.total_harm_score)
        results_right_model['pedestrians_hit'].append(len(scenario.collided_pedestrians))
        results_right_model['pedestrian_ages'].append(scenario.pedestrian_ages)
    
        if len(scenario.collided_pedestrians) > 0:
            collided_pedestrians = [x for x in scenario.collided_pedestrians]
            pedestrian_hit_ages = [scenario.pedestrian_attributes[collided_pedestrian]['age'] for collided_pedestrian in collided_pedestrians]
            results_right_model['pedestrian_hit_ages'].extend(pedestrian_hit_ages)
            max_age = max(max(results_right_model['pedestrian_ages']))
            min_age = min(min(results_right_model['pedestrian_ages']))

            try:
                normalized_age_hit = [(age - min_age) / (max_age - min_age) for age in pedestrian_hit_ages]
            
            except ZeroDivisionError:
                normalized_age_hit = [1 for age in pedestrian_hit_ages]

            scenario_age_hit = sum(normalized_age_hit) / len(normalized_age_hit)
            ages_hit.append(scenario_age_hit)

        else:
            ages_hit.append(2)

        results_right_model['harm_scores'] = []
        results_right_model['pedestrians_hit'] = []
        results_right_model['pedestrian_ages'] = []


        scenario.trivial_run('straight')


        results_straight_model['harm_scores'].append(scenario.total_harm_score)
        results_straight_model['pedestrians_hit'].append(len(scenario.collided_pedestrians))
        results_straight_model['pedestrian_ages'].append(scenario.pedestrian_ages)
    
        if len(scenario.collided_pedestrians) > 0:
            collided_pedestrians = [x for x in scenario.collided_pedestrians]
            pedestrian_hit_ages = [scenario.pedestrian_attributes[collided_pedestrian]['age'] for collided_pedestrian in collided_pedestrians]
            results_straight_model['pedestrian_hit_ages'].extend(pedestrian_hit_ages)
            max_age = max(max(results_straight_model['pedestrian_ages']))
            min_age = min(min(results_straight_model['pedestrian_ages']))

            try:
                normalized_age_hit = [(age - min_age) / (max_age - min_age) for age in pedestrian_hit_ages]
            
            except ZeroDivisionError:
                normalized_age_hit = [1 for age in pedestrian_hit_ages]

            scenario_age_hit = sum(normalized_age_hit) / len(normalized_age_hit)
            ages_hit.append(scenario_age_hit)

        else:
            ages_hit.append(2)

        results_straight_model['harm_scores'] = []
        results_straight_model['pedestrians_hit'] = []
        results_straight_model['pedestrian_ages'] = []
    fig, ax = plt.subplots()

    ax.hist(ages_hit, bins=15, linewidth=0.5, edgecolor="white")
    plt.show()

    
