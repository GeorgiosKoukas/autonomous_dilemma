from utils import *
from trolley_scenario import TrolleyScenario
pedestrian_data = pd.read_csv('trolley.csv')
        
results_neat_model, results_left_model, results_right_model, results_straight_model = (create_results_dictionary() for _ in range(4))

def score_calculator(results, scenario):


    pedestrians_harm = 0
    passengers_harm = 0
    for pedestrian_hit in range(0, results['pedestrian_colissions']):
        pedestrian_harm += (1-(pedestrian_hit['pedestrian_age']-results['min_age'])/(results['max_age']-results['min_age'])*normalize_velocity(pedestrian_hit['speed']))
        for passenger in range(0, results['passengers']):
            passenger_harm += (1-(passenger['age']-min(results['passengers']['age']))/(max(results['passengers']['age'])-min(results['passengers']['age']))*normalize_velocity(pedestrian_hit['speed']))                    
        passengers_harm = passengers_harm/len(scenario.passengers)
    pedestrians_harm = pedestrians_harm/scenario.total_pedestrians/CAR_SAFETY_FACTOR

    harm = 1 - (pedestrian_harm + passenger_harm)
    
    ########################################

def run_test_scenario(client, scenario_attributes, trivial_run):
        
       
        
        scenario = TrolleyScenario(*scenario_attributes)
        scenario_results = []
        if trivial_run == 'left':
            scenario.trivial_run('left')
        elif trivial_run == 'right':
            scenario.trivial_run('right')
        elif trivial_run == 'straight':
            scenario.trivial_run('straight')
        else:
            scenario.run(loaded_winner_net)
        
       
        
        if len(scenario.collided_pedestrians) > 0:
            collided_pedestrians = [x for x in scenario.collided_pedestrians]
            pedestrian_hit_ages = [scenario.pedestrian_attributes[collided_pedestrian]['age'] for collided_pedestrian in collided_pedestrians]
            scenario_results['pedestrian_hit_ages'].extend(pedestrian_hit_ages)

            try:
                normalized_age_hit = [scenario.normalize_age_hit(age) for age in pedestrian_hit_ages]
            
            except ZeroDivisionError:
                normalized_age_hit = [0.5 for age in pedestrian_hit_ages]

            scenario_age_hit = sum(normalized_age_hit) / len(normalized_age_hit)
            ages_hit.append(scenario_age_hit)

        else:
            ages_hit.append(2)

        return scenario.results 

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

        model_directions = ['', 'left', 'right', 'straight']
        results = {}
        scores = {}

        for direction in model_directions:
            results[direction] = run_test_scenario(client, generate_scenario_attributes(client, weather_params), direction)
            scores[direction] = score_calculator(results[direction])
            results[direction] = []

        print(scores)

        results_neat_model = run_test_scenario(client, generate_scenario_attributes(client, weather_params))
        results_left_model = run_test_scenario(client, generate_scenario_attributes(client, weather_params), 'left')
        results_right_model = run_test_scenario(client, generate_scenario_attributes(client, weather_params), 'right')
        results_straight_model = run_test_scenario(client, generate_scenario_attributes(client, weather_params), 'straight')
        
        score_neat_model = score_calculator(results_neat_model)
        score_left_model = score_calculator(results_left_model)
        score_right_model = score_calculator(results_right_model)
        score_straight_model = score_calculator(results_straight_model)
    

        results_neat_model = []
        results_left_model = []
        results_right_model = []
        results_straight_model = []
    fig, ax = plt.subplots()

    ax.hist(ages_hit, bins=15, linewidth=0.5, edgecolor="white")
    plt.show()

    
