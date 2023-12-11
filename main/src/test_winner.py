import pickle
import carla
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import *
from trolley_scenario import TrolleyScenario

def load_winner_net():
    with open("winner_net.pkl", "rb") as input_file:
        return pickle.load(input_file)

def run_test_scenario(client, scenario_attributes, trivial_run, loaded_winner_net):
    scenario = TrolleyScenario(*scenario_attributes)
    
    if trivial_run == "left":
        scenario.run(loaded_winner_net, controlling_driver="", choice="left")
    elif trivial_run == "right":
        scenario.run(loaded_winner_net, controlling_driver="", choice="right")
    elif trivial_run == "straight":
        scenario.run(loaded_winner_net, controlling_driver="", choice="straight")
    elif trivial_run == "no":
        scenario.run(loaded_winner_net, controlling_driver="neat", choice="neat")

    score = score_calculator(scenario.results, scenario)
    return scenario.results, score

def run_scenarios(client, num_scenarios, loaded_winner_net):
    scores = {'neat': [], 'left': [], 'right': [], 'straight': []}
    for _ in range(num_scenarios):
        attributes = generate_scenario_attributes(client)
        for run_type, choice in zip(scores.keys(), ["no", "left", "right", "straight"]):
            _, harm = run_test_scenario(client, attributes, choice, loaded_winner_net)
            scores[run_type].append(1 - harm)
    return scores
def plot_average_scores(scores):
    average_scores = {model: sum(score_list) / len(score_list) for model, score_list in scores.items()}
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(average_scores.keys()), y=list(average_scores.values()))
    plt.title('Average Scores of Models')
    plt.xlabel('Model')
    plt.ylabel('Average Score')
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
    plt.show()
def plot_score_distribution(scores):
    flattened_scores = [(model, score) for model, score_list in scores.items() for score in score_list]
    df = pd.DataFrame(flattened_scores, columns=['Model', 'Score'])
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Model', y='Score', data=df)
    plt.title('Score Distribution of Models')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.show()
def plot_heatmap(scores):
    score_matrix = pd.DataFrame(scores)
    plt.figure(figsize=(12, 8))
    sns.heatmap(score_matrix, annot=True, fmt=".2f", cmap='viridis')
    plt.title('Scenario-wise Scores of Models')
    plt.xlabel('Model')
    plt.ylabel('Scenario Number')
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
    plt.show()

def plot_violin_scores(scores):
    flattened_scores = [(model, score) for model, score_list in scores.items() for score in score_list]
    df = pd.DataFrame(flattened_scores, columns=['Model', 'Score'])
    
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Model', y='Score', data=df)
    plt.title('Score Distribution of Models Using Violin Plot')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.show()

if __name__ == "__main__":
    client = carla.Client("localhost", 2000)
    client.set_timeout(15)
    world = client.get_world()
    settings_setter(world)

    pedestrian_data = pd.read_csv("trolley.csv")
    loaded_winner_net = load_winner_net()
    num_scenarios = 300
    scores = run_scenarios(client, num_scenarios, loaded_winner_net)

    sns.set_style("dark")
    sns.set_context("poster")
    sns.set_palette("deep")

    plot_scores(scores)
    plot_average_scores(scores)
    plot_cumulative_scores(scores)
    plot_score_distribution(scores)
    plot_heatmap(scores)
    plot_violin_scores(scores)