from utils import *
from utils import NUM_EPISODES
from trolley_scenario import TrolleyScenario


def eval_genomes(genomes, config):
    client = carla.Client("localhost", 2000)
    client.set_timeout(15)
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True
    weather_params = {
        "cloudiness": 0.0,
        "precipitation": 50.0,
        "sun_altitude_angle": 90.0,
    }
    # settings.no_rendering_mode = True
    world.apply_settings(settings)

    generation_scenarios = []
    for scenario in range(NUM_EPISODES):
        groups_config = generate_groups_config(NUM_GROUPS)
        group_offsets = [
            set_random_offsets()[0] if i == 0 else set_random_offsets()[1]
            for i in range(len(groups_config["groups"]) + 1)
        ]
        total_pedestrians = sum([group["number"] for group in groups_config["groups"]])
        generation_scenarios.append(
            (
                groups_config,
                client,
                weather_params,
                pedestrian_data.sample(total_pedestrians).to_dict("records"),
                [
                    [generate_spawn_location() for _ in range(group["number"])]
                    for group in groups_config["groups"]
                ],
                group_offsets,
            )
        )

    for genome_id, genome in genomes:
        genome_fitness = []
        genome.fitness = 0
        for attributes in range(NUM_EPISODES):
            scenario_attributes = generation_scenarios[attributes]

            net = neat.nn.FeedForwardNetwork.create(genome, config)
            # Generate the same scenario for each AV in the same generation
            scenario = TrolleyScenario(*scenario_attributes)
            scenario.run(net, "neat", "no")
            # Use the results to determine the loss
            harm_score = MAGNYFYING_FITNESS*score_calculator(scenario.results, scenario)
            genome_fitness.append(-harm_score)
        genome.fitness = sum(genome_fitness)
        print(f"Genome {genome_id} fitness: {genome.fitness}")
