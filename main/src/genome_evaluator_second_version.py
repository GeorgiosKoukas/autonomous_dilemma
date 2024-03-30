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
        life = 0
        gone_right = False
        gone_left = False
        for attributes in range(NUM_EPISODES):
            scenario_attributes = generation_scenarios[attributes]
            # net = neat.nn.RecurrentNetwork.create(genome, config)
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            scenario = TrolleyScenario(*scenario_attributes)
            scenario.run(net, "neat", "no")
            harm_score, _, _ = score_calculator(scenario.results, scenario)
            harm_score = MAGNYFYING_FITNESS * harm_score
            genome_fitness.append(-harm_score)
            turn = sum(scenario.steering)

            if turn > 0:
                gone_right = True

            if turn < 0:
                gone_left = True

            if len(scenario.collided_pedestrians) == 0:
                genome_fitness.append(100)
        for attributes in range(NUM_EPISODES, NUM_MAX_EPISODES):
            scenario_attributes = generation_scenarios[attributes]

            net = neat.nn.FeedForwardNetwork.create(genome, config)
            scenario = TrolleyScenario(*scenario_attributes)
            scenario.run(net, "neat", "no")

            harm_score, _, _ = score_calculator(scenario.results, scenario)
            harm_score = MAGNYFYING_FITNESS * harm_score
            turn = sum(scenario.steering)
            if turn > 0:
                gone_right = True
            if turn < 0:
                gone_left = True
            if attributes > 1:
                if not gone_left or not gone_right:
                    genome_fitness.append(-3000)
                    break
            if attributes % 10 == 0:
                life += 1
                print(f"life gained, reached scenario {attributes} with life {life}")

            if abs(harm_score) > 0:
                genome_fitness.append(-harm_score)
                if life > 0:
                    print(
                        f"life lost, reached scenario {attributes} with life {life-1}"
                    )
                life = life - 1

                if life < 0 and attributes > 1:
                    break
            else:
                genome_fitness.append(100)

        genome.fitness = sum(genome_fitness)
        del scenario
        if genome.fitness > 2500:
            save_dir = "saved_genomes"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        
            filename = os.path.join(
                save_dir, f"genome_{genome_id}_fitness_{genome.fitness}.pkl"
            )

            # Dump the genome using pickle
            with open(filename, "wb") as f:
                pickle.dump(genome, f)
        print(f"Genome {genome_id} fitness: {genome.fitness}")
