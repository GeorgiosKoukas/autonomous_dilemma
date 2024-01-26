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
filepath = "saved_genomes/golden ones/genome_25664_fitness_9209.538459962874.pkl"
with open(filepath, "rb") as input_file:
        winner_net = pickle.load(input_file)
        

            
visualize.draw_net(config, winner_net, True)


   





config = ConfigObj(config_path, write_empty_values=True)
num_inputs = NUM_GROUPS * MAX_PEDS * 3 + 2 + 1
config["DefaultGenome"]["num_inputs"] = num_inputs
config["DefaultGenome"]["num_hidden"] = int(0.8 * num_inputs)
config.write()  # node response options

run(config_path)
