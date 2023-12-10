from utils import *
from trolley_scenario import TrolleyScenario

pedestrian_data = pd.read_csv("trolley.csv")

results_neat_model, results_left_model, results_right_model, results_straight_model = (
    create_results_dictionary() for _ in range(4)
)

def run_test_scenario(client, scenario_attributes, trivial_run):
    scenario = TrolleyScenario(*scenario_attributes)
    
    if trivial_run == "left":
        scenario.run(loaded_winner_net, controlling_driver = "", choice="left")
    elif trivial_run == "right":
        scenario.run(loaded_winner_net, controlling_driver = "", choice="right")
    elif trivial_run == "straight":
        scenario.run(loaded_winner_net, controlling_driver = "", choice="straight")
    elif trivial_run == "no":
        scenario.run(loaded_winner_net, controlling_driver = "neat", choice="neat")

    # if len(scenario.collided_pedestrians) > 0:
    #     collided_pedestrians = [x for x in scenario.collided_pedestrians]
    #     pedestrian_hit_ages = [
    #         scenario.pedestrian_attributes[collided_pedestrian]["age"]
    #         for collided_pedestrian in collided_pedestrians
    #     ]
    #     scenario_results["pedestrian_hit_ages"].extend(pedestrian_hit_ages)

    #     try:
    #         normalized_age_hit = [
    #             scenario.normalize_age(age) for age in pedestrian_hit_ages
    #         ]

    #     except ZeroDivisionError:
    #         normalized_age_hit = [0.5 for age in pedestrian_hit_ages]

    #     scenario_age_hit = sum(normalized_age_hit) / len(normalized_age_hit)
    #     ages_hit.append(scenario_age_hit)

    # else:
    #     ages_hit.append(2)
    score = score_calculator(scenario.results, scenario)
    return scenario.results, score


if __name__ == "__main__":
    with open("winner_net.pkl", "rb") as input_file:
        loaded_winner_net = pickle.load(input_file)

    client = carla.Client("localhost", 2000)
    client.set_timeout(15)
    world = client.get_world()
    settings_setter(world)
    ages_hit = []
    num_scenarios = 100  # or any number of scenarios for every car
    scores_neat_model = []
    scores_left_model = []
    scores_right_model = []
    scores_straight_model = []
    for _ in range(num_scenarios):
        model_directions = ["", "left", "right", "straight"]
        results = {}
        scores = {}
        attributes = generate_scenario_attributes(client)
        # for direction in model_directions:
        #     results[direction] = run_test_scenario(client, attributes, direction)
        #     scores[direction] = score_calculator(results[direction])
        #     results[direction] = []

        # print(scores)

        results_neat_model, score_neat_model = run_test_scenario(
            client, generate_scenario_attributes(client), trivial_run = "no"
        )
        results_left_model, score_left_model = run_test_scenario(
            client, generate_scenario_attributes(client), trivial_run = "left"
        )
        results_right_model, score_right_model = run_test_scenario(
            client, generate_scenario_attributes(client), trivial_run = "right"
        )
        results_straight_model, score_straight_model = run_test_scenario(
            client, generate_scenario_attributes(client), trivial_run = "straight"
        )

        scores_neat_model.append(score_neat_model)
        scores_left_model.append(score_left_model)
        scores_right_model.append(score_right_model)
        scores_straight_model.append(score_straight_model)

        results_neat_model = []
        results_left_model = []
        results_right_model = []
        results_straight_model = []

    print(f"Neat model average score: {sum(scores_neat_model) / len(scores_neat_model)}")
    print(f"Left model average score: {sum(scores_left_model) / len(scores_left_model)}")
    print(f"Right model average score: {sum(scores_right_model) / len(scores_right_model)}")
    print(f"Straight model average score: {sum(scores_straight_model) / len(scores_straight_model)}")

        # Number of scenarios - should match the length of your score lists
    num_scenarios = len(scores_neat_model)

    # Generate x-axis values (scenario indices)
    scenarios = list(range(1, num_scenarios + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(scenarios, scores_neat_model, label='NEAT Model', marker='o')
    plt.plot(scenarios, scores_left_model, label='Left Model', marker='x')
    plt.plot(scenarios, scores_right_model, label='Right Model', marker='^')
    plt.plot(scenarios, scores_straight_model, label='Straight Model', marker='s')

    plt.title('Model Score Comparison')
    plt.xlabel('Scenario Number')
    plt.ylabel('Score')
    plt.xticks(scenarios)  # Set x-axis ticks to match the number of scenarios
    plt.legend()
    plt.grid(True)
    plt.show()
