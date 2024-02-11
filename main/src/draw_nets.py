from utils import *
from genome_evaluator import eval_genomes
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, "config.txt")
config = neat.config.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    config_path,
)
node_names = generate_node_names(MAX_PEDS, NUM_GROUPS)
filepath = "saved_genomes/genome_9411_fitness_9029.82748334049.pkl"
with open(filepath, "rb") as input_file:
        winner_net = pickle.load(input_file)
                    
visualize.draw_net(config, winner_net, True)

