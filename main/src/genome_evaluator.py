from utils import *
from utils import NUM_EPISODES
from trolley_scenario import TrolleyScenario


def eval_genomes(genomes, config):
    client = carla.Client("localhost", 2000)
    client.set_timeout(15)
    world = client.get_world()
    settings_setter(world)
    
    generation_scenarios = []
    for scenario in range(NUM_MAX_EPISODES):
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
            if len(scenario.collided_pedestrians) == 0:
                genome_fitness.append(100)
        if sum(genome_fitness)>0:
            for attributes in range(NUM_EPISODES, NUM_MAX_EPISODES):
                scenario_attributes = generation_scenarios[attributes]

                net = neat.nn.FeedForwardNetwork.create(genome, config)
                # Generate the same scenario for each AV in the same generation
                scenario = TrolleyScenario(*scenario_attributes)
                scenario.run(net, "neat", "no")
                # Use the results to determine the loss
                
                harm_score = MAGNYFYING_FITNESS*score_calculator(scenario.results, scenario)
                if abs(harm_score)>0:
                    genome_fitness.append(-harm_score)
                    break
                genome_fitness.append(100)

        genome.fitness = sum(genome_fitness)
        if genome.fitness > 900:
            # Specify a directory to save the genomes
            save_dir = "saved_genomes"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # Construct a file name based on the genome ID and fitness
            filename = os.path.join(save_dir, f"genome_{genome_id}_fitness_{genome.fitness}.pkl")

            # Dump the genome using pickle
            with open(filename, 'wb') as f:
                pickle.dump(genome, f)
        print(f"Genome {genome_id} fitness: {genome.fitness}")
    
