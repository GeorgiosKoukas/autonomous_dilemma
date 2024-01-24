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
    harm, normalized_pedestrian_harm, normalized_passenger_harm = score_calculator(scenario.results, scenario)
    return harm, normalized_pedestrian_harm, normalized_passenger_harm, scenario.elapsed_time_for_user_reaction

def plot_average_scores(scores, score_type):
    average_scores = {model: sum(score_list) / len(score_list) for model, score_list in scores.items()}
    if score_type == "pedestrian_harm":
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(average_scores.keys()), y=list(average_scores.values()))
        plt.title('Pedestrian Harm - Average Scores of Models')
        plt.xlabel('Model')
        plt.ylabel('Average Score')
        plt.savefig('plots/pedestrian_harm_scores/average_scores.png')
        plt.show()
    elif score_type == "passenger_harm":
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(average_scores.keys()), y=list(average_scores.values()))
        plt.title('Passenger Harm - Average Scores of Models')
        plt.xlabel('Model')
        plt.ylabel('Average Score')
        plt.savefig('plots/passengers_harm_scores/average_scores.png')
        plt.show()
    else:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(average_scores.keys()), y=list(average_scores.values()))
        plt.title('Average Scores of Models')
        plt.xlabel('Model')
        plt.ylabel('Average Score')
        plt.savefig('plots/overall_scores(with Ethical Knob)/average_scores.png')
        plt.show()


def plot_cumulative_scores(scores, score_type):
    """
    Plots the cumulative scores of different models based on the given score type.

    Parameters:
    scores (dict): A dictionary containing the scores of different models.
    score_type (str): The type of score to plot. Can be "pedestrian_harm", "passenger_harm", or any other type.

    Returns:
    None
    """
    if score_type == "pedestrian_harm":
        plt.figure(figsize=(12, 6))
        for model, score_list in scores.items():
            cumulative_scores = [sum(score_list[:i+1]) for i in range(len(score_list))]
            sns.lineplot(x=range(1, len(score_list) + 1), y=cumulative_scores, label=f'{model.capitalize()} Model')
        plt.title('Pedestrian Harm - Cumulative Scores of Models')
        plt.xlabel('Scenario Number')
        plt.ylabel('Cumulative Score')
        plt.legend()
        plt.savefig('plots/pedestrian_harm_scores/cumulative_scores.png')
        plt.show()
    elif score_type == "passenger_harm":
        plt.figure(figsize=(12, 6))
        for model, score_list in scores.items():
            cumulative_scores = [sum(score_list[:i+1]) for i in range(len(score_list))]
            sns.lineplot(x=range(1, len(score_list) + 1), y=cumulative_scores, label=f'{model.capitalize()} Model')
        plt.title('Passenger Harm - Cumulative Scores of Models')
        plt.xlabel('Scenario Number')
        plt.ylabel('Cumulative Score')
        plt.legend()
        plt.savefig('plots/passengers_harm_scores/cumulative_scores.png')
        plt.show()
    else:
        plt.figure(figsize=(12, 6))
        for model, score_list in scores.items():
            cumulative_scores = [sum(score_list[:i+1]) for i in range(len(score_list))]
            sns.lineplot(x=range(1, len(score_list) + 1), y=cumulative_scores, label=f'{model.capitalize()} Model')
        plt.title('Cumulative Scores of Models')
        plt.xlabel('Scenario Number')
        plt.ylabel('Cumulative Score')
        plt.legend()
        plt.savefig('plots/overall_scores(with Ethical Knob)/cumulative_scores.png')
        plt.show()

def plot_score_distribution(scores, score_type):
    flattened_scores = [(model, score) for model, score_list in scores.items() for score in score_list]
    df = pd.DataFrame(flattened_scores, columns=['Model', 'Score'])
    
    if score_type == "pedestrian_harm":
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Model', y='Score', data=df)
        plt.title('Pedestrian Harm - Score Distribution of Models')
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.savefig('plots/pedestrian_harm_scores/score_distribution.png')
        plt.show()
    elif score_type == "passenger_harm":
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Model', y='Score', data=df)
        plt.title('Passenger Harm - Score Distribution of Models')
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.savefig('plots/passengers_harm_scores/score_distribution.png')
        plt.show()
    else:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Model', y='Score', data=df)
        plt.title('Score Distribution of Models')
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.savefig('plots/overall_scores(with Ethical Knob)/score_distribution.png')
        plt.show()

def plot_heatmap(scores, score_type):
    
    score_matrix = pd.DataFrame(scores)
    if score_type == "pedestrian_harm":
        plt.figure(figsize=(12, 8))
        sns.heatmap(score_matrix, annot=True, fmt=".2f", cmap='viridis')
        plt.title('Pedestrian Harm - Scenario-wise Scores of Models')
        plt.xlabel('Model')
        plt.ylabel('Scenario Number')
        plt.savefig('plots/pedestrian_harm_scores/heatmap.png')
        plt.show()
    elif score_type == "passenger_harm":
        plt.figure(figsize=(12, 8))
        sns.heatmap(score_matrix, annot=True, fmt=".2f", cmap='viridis')
        plt.title('Passenger Harm - Scenario-wise Scores of Models')
        plt.xlabel('Model')
        plt.ylabel('Scenario Number')
        plt.savefig('plots/passengers_harm_scores/heatmap.png')
        plt.show()
    else:
        plt.figure(figsize=(12, 8))
        sns.heatmap(score_matrix, annot=True, fmt=".2f", cmap='viridis')
        plt.title('Scenario-wise Scores of Models')
        plt.xlabel('Model')
        plt.ylabel('Scenario Number')
        plt.savefig('plots/overall_scores(with Ethical Knob)/heatmap.png')
        plt.show()

def plot_scores(scores, score_type):
    sns.set(style="whitegrid")
    if score_type == "pedestrian_harm":
        plt.figure(figsize=(12, 6))
        for model, score_list in scores.items():
            sns.lineplot(x=range(1, len(score_list) + 1), y=score_list, label=f'{model.capitalize()} Model')
        
        plt.title('Pedestrian Harm - Model Score Comparison')
        plt.xlabel('Scenario Number')
        plt.ylabel('Score')
        plt.legend()
        plt.savefig('plots/pedestrian_harm_scores/scores.png')
        plt.show()
    elif score_type == "passenger_harm":
        plt.figure(figsize=(12, 6))
        for model, score_list in scores.items():
            sns.lineplot(x=range(1, len(score_list) + 1), y=score_list, label=f'{model.capitalize()} Model')
        
        plt.title('Passenger Harm - Model Score Comparison')
        plt.xlabel('Scenario Number')
        plt.ylabel('Score')
        plt.legend()
        plt.savefig('plots/passengers_harm_scores/scores.png')
        plt.show()
    else:
        plt.figure(figsize=(12, 6))

        for model, score_list in scores.items():
            sns.lineplot(x=range(1, len(score_list) + 1), y=score_list, label=f'{model.capitalize()} Model')

        plt.title('Model Score Comparison')
        plt.xlabel('Scenario Number')
        plt.ylabel('Score')
        plt.legend()
        plt.savefig('plots/overall_scores(with Ethical Knob)/scores.png')
        plt.show()

def plot_violin_scores(scores, score_type):
    flattened_scores = [(model, score) for model, score_list in scores.items() for score in score_list]
    df = pd.DataFrame(flattened_scores, columns=['Model', 'Score'])
    if score_type == "pedestrian_harm":
        plt.figure(figsize=(10, 6))
        sns.violinplot(x='Model', y='Score', data=df)
        plt.title('Pedestrian Harm - Score Distribution of Models Using Violin Plot')
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.savefig('plots/pedestrian_harm_scores/violin_scores.png')
        plt.show()
    elif score_type == "passenger_harm":
        plt.figure(figsize=(10, 6))
        sns.violinplot(x='Model', y='Score', data=df)
        plt.title('Passenger Harm - Score Distribution of Models Using Violin Plot')
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.savefig('plots/passengers_harm_scores/violin_scores.png')
        plt.show()
    else:
        plt.figure(figsize=(10, 6))
        sns.violinplot(x='Model', y='Score', data=df)
        plt.title('Score Distribution of Models Using Violin Plot')
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.savefig('plots/overall_scores(with Ethical Knob)/violin_scores.png')
        plt.show()

def run_scenarios(client, num_scenarios, winners, choices):
    scores = {choice: [] for choice in choices}
    scores_pedestrian_harm = {choice: [] for choice in choices}
    scores_passenger_harm = {choice: [] for choice in choices}
    reaction_times = []
    for _ in range(num_scenarios):
        attributes = generate_scenario_attributes(client)
        scores_per_scenario = []
        for choice in choices:
            if choice != "manual":
                winner = winners.get(choice, None)
                harm, normalized_pedestrian_harm, normalized_passenger_harm, _ = run_test_scenario( attributes, choice, winner)
                score = 1 - harm
                scores[choice].append(score)
                scores_per_scenario.append(score)
                score_pedestrian_harm = 1 - normalized_pedestrian_harm
                scores_pedestrian_harm[choice].append(score_pedestrian_harm)
                score_passenger_harm = 1 - normalized_passenger_harm
                scores_passenger_harm[choice].append(score_passenger_harm)
            else:
                harm, normalized_pedestrian_harm, normalized_passenger_harm, reaction_time = run_test_scenario( attributes, choice, None)
                score = 1 - harm
                scores[choice].append(score)
                scores_per_scenario.append(score)
                score_pedestrian_harm = 1 - normalized_pedestrian_harm
                scores_pedestrian_harm[choice].append(score_pedestrian_harm)
                score_passenger_harm = 1 - normalized_passenger_harm
                scores_passenger_harm[choice].append(score_passenger_harm)
                reaction_times.append(reaction_time)
       


        max_score = max(scores_per_scenario)
        min_score = min(scores_per_scenario)

        # Normalize and add scores to the dictionary
        for run_type, score in zip(choices, scores_per_scenario):
            try:
                normalized_score = (score - min_score) / (max_score - min_score)
            except ZeroDivisionError:
                normalized_score = 0.5  # Handle division by zero if max_score equals min_score
            scores[run_type].append(normalized_score)

    return scores, scores_pedestrian_harm, scores_passenger_harm, reaction_times

def run_scenarios_user_input(client, num_scenarios, winners, choices):
    scores = {choice: [] for choice in choices}
    scores_pedestrian_harm = {choice: [] for choice in choices}
    scores_passenger_harm = {choice: [] for choice in choices}

    for _ in range(num_scenarios):
        attributes = generate_scenario_attributes(client)
        scores_per_scenario = []
        for choice in choices:
            winner = winners.get(choice, None)
            harm, normalized_pedestrian_harm, normalized_passenger_harm = run_test_scenario( attributes, choice, winner)
            score = 1 - harm
            scores[choice].append(score)
            scores_per_scenario.append(score)
            score_pedestrian_harm = 1 - normalized_pedestrian_harm
            scores_pedestrian_harm[choice].append(score_pedestrian_harm)
            score_passenger_harm = 1 - normalized_passenger_harm
            scores_passenger_harm[choice].append(score_passenger_harm)
       


        max_score = max(scores_per_scenario)
        min_score = min(scores_per_scenario)

        # Normalize and add scores to the dictionary
        for run_type, score in zip(choices, scores_per_scenario):
            try:
                normalized_score = (score - min_score) / (max_score - min_score)
            except ZeroDivisionError:
                normalized_score = 0.5  # Handle division by zero if max_score equals min_score
            scores[run_type].append(normalized_score)

    return scores, scores_pedestrian_harm, scores_passenger_harm

if __name__ == "__main__":
    client = carla.Client("localhost", 2000)
    client.set_timeout(15)
    world = client.get_world()
    settings_setter(world)

    filepaths = [
               "saved_genomes/golden ones/genome_25664_fitness_9209.538459962874.pkl"

    ]

    # folder_path = 'saved_genomes'


    # filepaths = []

    # for file in os.listdir(folder_path):

    #     file_path = os.path.join(folder_path, file)
    #     if os.path.isfile(file_path):

    #         filepaths.append(file_path)




    winners = load_winner_nets(filepaths)

    num_scenarios = 10
    choices = list(winners.keys()) + ["left", "right", "straight", "manual"] # Add or remove choices as needed
    choices = ["manual"]
    overall_scores, pedestrian_scores, passenger_scores, reaction_times = run_scenarios(client, num_scenarios, winners, choices)
    plt.figure()
    sns.distplot(reaction_times)
    plt.title('Reaction Times of Manual Driver')
    plt.xlabel('Reaction Time (s)')
    plt.ylabel('Density')
    plt.savefig('plots/reaction_times.png')
    plt.show()
    plot_average_scores(overall_scores, None)
    plot_cumulative_scores(overall_scores, None)
    plot_score_distribution(overall_scores, None)
    plot_heatmap(overall_scores, None)
    plot_scores(overall_scores, None)
    plot_violin_scores(overall_scores, None)

    plot_average_scores(pedestrian_scores, "pedestrian_harm")
    plot_cumulative_scores(pedestrian_scores, "pedestrian_harm")
    plot_score_distribution(pedestrian_scores, "pedestrian_harm")
    plot_heatmap(pedestrian_scores, "pedestrian_harm")
    plot_scores(pedestrian_scores, "pedestrian_harm")
    plot_violin_scores(pedestrian_scores, "pedestrian_harm")

    plot_average_scores(passenger_scores, "passenger_harm")
    plot_cumulative_scores(passenger_scores, "passenger_harm")
    plot_score_distribution(passenger_scores, "passenger_harm")
    plot_heatmap(passenger_scores, "passenger_harm")
    plot_scores(passenger_scores, "passenger_harm")
    plot_violin_scores(passenger_scores, "passenger_harm")



    
