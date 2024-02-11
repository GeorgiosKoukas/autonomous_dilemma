import neat
import gzip
import pandas as pd
import matplotlib.pyplot as plt

NUMBER_OF_GENERATIONS = 1000  # Adjust based on your number of generations

def load_checkpoint(filename):
    checkpoint = neat.Checkpointer(1, filename_prefix="neat-checkpoint-")
    p = checkpoint.restore_checkpoint(filename)
    return p

def extract_species_data(p):
    species_sizes = {}
    for sid, species in p.species.species.items():
        species_sizes[sid] = len(species.members)
    return species_sizes

def calculate_max_depth(genome, config):
    # Initialize all node depths to 0
    node_depths = {node_key: 0 for node_key in genome.nodes.keys()}

    # Continuously update depths until no changes occur
    changed = True
    while changed:
        changed = False
        for (in_node, out_node), cg in genome.connections.items():
            if cg.enabled and in_node in node_depths and out_node in node_depths:
                depth = node_depths[in_node] + 1
                if depth > node_depths[out_node]:
                    node_depths[out_node] = depth
                    changed = True

    return max(node_depths.values())




species_data = {}
best_fitness_per_generation = []
topology_data = {'generation': [], 'avg_nodes': [], 'avg_connections': []}
max_depth_per_generation = []

for i in range(NUMBER_OF_GENERATIONS):  # Looping through checkpoints
    filename = f"checkpoints/neat-checkpoint-{i}"
    p = load_checkpoint(filename)

    # Speciation Data
    species_sizes = extract_species_data(p)
    species_data[i] = species_sizes

    # Best Fitness Data
    best_fitness = max([g.fitness for g in p.population.values() if g.fitness is not None], default=0)
    best_fitness_per_generation.append(best_fitness)

    # Topological Complexity Data
    total_nodes = 0
    total_connections = 0
    num_genomes = len(p.population)

    for genome_id, genome in p.population.items():
        total_nodes += len(genome.nodes)
        total_connections += len(genome.connections)

    avg_nodes = total_nodes / num_genomes
    avg_connections = total_connections / num_genomes

    topology_data['generation'].append(i)
    topology_data['avg_nodes'].append(avg_nodes)
    topology_data['avg_connections'].append(avg_connections)

    # Approximated Hidden Layer Depth
    generation_max_depth = 0
    for genome_id, genome in p.population.items():
        depth = calculate_max_depth(genome, p.config)  # Implement this function
        generation_max_depth = max(generation_max_depth, depth)
    max_depth_per_generation.append(generation_max_depth)

# Convert topological data to DataFrame
df_topology = pd.DataFrame(topology_data)

# Speciation Plot
stacked_data = []
for i in range(NUMBER_OF_GENERATIONS):
    for sid in species_data[i]:
        stacked_data.append({
            'Generation': i,
            'Species': sid,
            'Genomes': species_data[i][sid]
        })

df_stacked = pd.DataFrame(stacked_data)
df_pivot = df_stacked.pivot(index='Generation', columns='Species', values='Genomes').fillna(0)

plt.figure(figsize=(12, 6))
plt.stackplot(df_pivot.index, df_pivot.T, labels=df_pivot.columns)
plt.xlabel('Generation')
plt.ylabel('Number of Genomes per Species')
plt.title('Genome Distribution in Species over Generations')
#plt.legend(loc='upper left')
plt.savefig('plots/training_progress/species_distribution.png')
plt.show()

# Best Fitness Plot
plt.figure(figsize=(12, 6))
plt.plot(range(NUMBER_OF_GENERATIONS), best_fitness_per_generation, color='red', label='Best Fitness')
plt.xlabel('Generation')
plt.ylabel('Best Fitness Score')
plt.title('Best Fitness over Generations')
plt.legend()
plt.savefig('plots/training_progress/best_fitness.png')
plt.show()

# Topological Complexity Plot
plt.figure(figsize=(12, 6))
plt.plot(df_topology['generation'], df_topology['avg_nodes'], label='Average Nodes')
plt.plot(df_topology['generation'], df_topology['avg_connections'], label='Average Connections')
plt.xlabel('Generation')
plt.ylabel('Average Count')
plt.title('Topological Complexity over Generations')
plt.legend()
plt.savefig('plots/training_progress/topological_complexity.png')
plt.show()

# Hidden Layers Approximation Plot
plt.figure(figsize=(12, 6))
plt.plot(range(NUMBER_OF_GENERATIONS), max_depth_per_generation, label='Approximated Hidden Layers')
plt.xlabel('Generation')
plt.ylabel('Max Depth (Hidden Layers)')
plt.title('Approximation of Hidden Layers over Generations')
plt.legend()
plt.savefig('plots/training_progress/hidden_layers_approximation.png')
plt.show()
