import os
import pickle
import neat
import carla
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import *
from trolley_scenario import TrolleyScenario

def load_winner_nets(filepaths):
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )
    
    winners = {}
    for idx, filepath in enumerate(filepaths):
        with open(filepath, "rb") as input_file:
            winner_net = neat.nn.FeedForwardNetwork.create(pickle.load(input_file), config)
            winners[f"neat{idx+1}"] = winner_net
    return winners

def run_test_scenario(scenario_attributes, trivial_run, winner):
    scenario = TrolleyScenario(*scenario_attributes)
    scenario.run(winner, controlling_driver=trivial_run if trivial_run.startswith("neat") else "", choice=trivial_run)
    score = score_calculator(scenario.results, scenario)
    return scenario.results, score

def plot_average_scores(scores):
    average_scores = {model: sum(score_list) / len(score_list) for model, score_list in scores.items()}
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(average_scores.keys()), y=list(average_scores.values()))
    plt.title('Average Scores of Models')
    plt.xlabel('Model')
    plt.ylabel('Average Score')
    plt.savefig('plots/average_scores.png')
    plt.show()


def plot_cumulative_scores(scores):
    plt.figure(figsize=(12, 6))
    for model, score_list in scores.items():
        cumulative_scores = [sum(score_list[:i+1]) for i in range(len(score_list))]
        sns.lineplot(x=range(1, len(score_list) + 1), y=cumulative_scores, label=f'{model.capitalize()} Model')
    
    plt.title('Cumulative Scores of Models')
    plt.xlabel('Scenario Number')
    plt.ylabel('Cumulative Score')
    plt.legend()
    plt.savefig('plots/cumulative_scores.png')
    plt.show()

def plot_score_distribution(scores):
    flattened_scores = [(model, score) for model, score_list in scores.items() for score in score_list]
    df = pd.DataFrame(flattened_scores, columns=['Model', 'Score'])
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Model', y='Score', data=df)
    plt.title('Score Distribution of Models')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.savefig('plots/score_distribution.png')
    plt.show()

def plot_heatmap(scores):
    score_matrix = pd.DataFrame(scores)
    plt.figure(figsize=(12, 8))
    sns.heatmap(score_matrix, annot=True, fmt=".2f", cmap='viridis')
    plt.title('Scenario-wise Scores of Models')
    plt.xlabel('Model')
    plt.ylabel('Scenario Number')
    plt.savefig('plots/heatmap.png')
    plt.show()

def plot_scores(scores):
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))

    for model, score_list in scores.items():
        sns.lineplot(x=range(1, len(score_list) + 1), y=score_list, label=f'{model.capitalize()} Model')

    plt.title('Model Score Comparison')
    plt.xlabel('Scenario Number')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig('plots/scores.png')
    plt.show()

def plot_violin_scores(scores):
    flattened_scores = [(model, score) for model, score_list in scores.items() for score in score_list]
    df = pd.DataFrame(flattened_scores, columns=['Model', 'Score'])
    
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Model', y='Score', data=df)
    plt.title('Score Distribution of Models Using Violin Plot')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.savefig('plots/violin_scores.png')
    plt.show()

def run_scenarios(client, num_scenarios, winners, choices):
    scores = {choice: [] for choice in choices}

    for _ in range(num_scenarios):
        attributes = generate_scenario_attributes(client)
        scores_per_scenario = []
        for choice in choices:
            winner = winners.get(choice, None)
            _, harm = run_test_scenario( attributes, choice, winner)
            score = 1 - harm
            scores[choice].append(score)
            scores_per_scenario.append(score)

        max_score = max(scores_per_scenario)
        min_score = min(scores_per_scenario)

        # Normalize and add scores to the dictionary
        for run_type, score in zip(choices, scores_per_scenario):
            try:
                normalized_score = (score - min_score) / (max_score - min_score)
            except ZeroDivisionError:
                normalized_score = 0.5  # Handle division by zero if max_score equals min_score
            scores[run_type].append(normalized_score)
    return scores



if __name__ == "__main__":
    client = carla.Client("localhost", 2000)
    client.set_timeout(15)
    world = client.get_world()
    settings_setter(world)

    filepaths = [
                #  "saved_genomes/genome_14969_fitness_1159.835511544773.pkl",
                "saved_genomes/genome_480_fitness_1183.4752340246591.pkl"
                #  "saved_genomes/genome_321_fitness_1259.8341205684658.pkl",
                #  "saved_genomes/genome_456_fitness_1164.272206410602.pkl"







                ]
    winners = load_winner_nets(filepaths)

    num_scenarios = 100
    choices = list(winners.keys()) + ["left", "right", "straight"] # Add or remove choices as needed
    scores = run_scenarios(client, num_scenarios, winners, choices)
    plot_average_scores(scores)
    plot_cumulative_scores(scores)
    plot_score_distribution(scores)
    plot_heatmap(scores)
    plot_scores(scores)
    plot_violin_scores(scores)


    
