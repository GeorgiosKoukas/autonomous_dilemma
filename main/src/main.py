from utils import *
from genome_evaluator import eval_genomes


def run(config_path):
    checkpoint_restorer = True
    checkpoint_restorer = True
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )
    output = "winner_net.pkl"
    if checkpoint_restorer:
        checkpoint = neat.Checkpointer(1, filename_prefix="neat-checkpoint-")

        p = checkpoint.restore_checkpoint("neat-checkpoint-734")
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        p.add_reporter(checkpoint)
       
 
        winner = p.run(eval_genomes, NUM_GENERATIONS)
        node_names = generate_node_names(MAX_PEDS, NUM_GROUPS)
        visualize.draw_net(config, winner, True, node_names=node_names)
        visualize.plot_stats(stats, ylog=False, view=True)
        visualize.plot_species(stats, view=True)
        winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
        pickle.dump(winner_net, output, pickle.HIGHEST_PROTOCOL)

            
    else:
        p = neat.Population(config)
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)

        checkpoint = neat.Checkpointer(1, filename_prefix="neat-checkpoint-")
        p.add_reporter(checkpoint)
      
        winner = p.run(eval_genomes, NUM_GENERATIONS)
        node_names = generate_node_names(MAX_PEDS, NUM_GROUPS)
        visualize.draw_net(config, winner, True, node_names=node_names)
        visualize.plot_stats(stats, ylog=False, view=True)
        visualize.plot_species(stats, view=True)
        winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
        pickle.dump(winner_net, output, pickle.HIGHEST_PROTOCOL)

    


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")

    # Update the value of num_inputs
    config = ConfigObj(config_path, write_empty_values=True)
    num_inputs = NUM_GROUPS * MAX_PEDS * 3 + 2 + 1
    config["DefaultGenome"]["num_inputs"] = num_inputs
    config["DefaultGenome"]["num_hidden"] = int(0.8*num_inputs)
    config.write()# node response options


    run(config_path)
