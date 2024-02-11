import os
import pickle
import neat
import carla
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import *
from trolley_scenario import TrolleyScenario
import squarify


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
            winner_net = neat.nn.FeedForwardNetwork.create(
                pickle.load(input_file), config
            )
            winners[f"neat{idx+1}"] = winner_net
    return winners


def run_test_scenario(scenario_attributes, trivial_run, winner):
    scenario = TrolleyScenario(*scenario_attributes)

    scenario.run(
        winner,
        controlling_driver=trivial_run if trivial_run.startswith("neat") else "",
        choice=trivial_run,
    )
    harm, normalized_pedestrian_harm, normalized_passenger_harm = score_calculator(
        scenario.results, scenario
    )
    return (
        harm,
        normalized_pedestrian_harm,
        normalized_passenger_harm,
        scenario.elapsed_time_for_user_reaction,
        sum(scenario.steering),
        scenario.steering,
    )


def plot_steering_traces(scenario_steering_data):
    plt.figure(figsize=(12, 6))

    for scenario_id, steering_values in scenario_steering_data.items():
        plt.plot(steering_values, label=f"Scenario {scenario_id}")

    plt.xlabel("Ticks")
    plt.ylabel("Steering Angle")
    plt.title("Steering Traces for All Scenarios")

    plt.savefig("plots/steering_traces.png")
    plt.close()


def plot_tree_map(turns_per_scenario):
    turns_per_scenario = turns_per_scenario[next(iter(turns_per_scenario))]
    print(turns_per_scenario)
    decisions = {"left": 0, "right": 0}
    for index, sum_of_turns in enumerate(turns_per_scenario):
        if sum_of_turns <= 1:
            decisions["left"] += 1
        else:
            decisions["right"] += 1
    print(decisions)
    labels = [f"{key}\n{value}" for key, value in decisions.items()]
    sizes = decisions.values()
    colors = [plt.cm.Spectral(i / float(len(labels))) for i in range(len(labels))]

    # Create a treemap
    plt.figure(figsize=(12, 8))
    squarify.plot(sizes=sizes, label=labels, color=colors, alpha=0.7)
    plt.title("Treemap of Decisions Made by a Specific Model")
    plt.axis("off")



def plot_average_scores(scores, score_type):
    average_scores = {
        model: sum(score_list) / len(score_list) for model, score_list in scores.items()
    }

    # Setting a color palette

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=list(average_scores.keys()), y=list(average_scores.values()))

    # Adding value annotations on each bar
    for p in ax.patches:
        ax.annotate(
            format(p.get_height(), ".2f"),
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 9),
            textcoords="offset points",
        )


    plt.title(f"{score_type} - Average Scores of Models")
    plt.ylabel("Average Score")

    plt.xlabel("Model")
    plt.xticks(rotation=45)  # Rotating model names for better visibility
    plt.tight_layout()  # Adjust layout
    # Save plot with a higher resolution
    plt.savefig(f"plots/{score_type}/average_scores.png", dpi=300)
    plt.close()


def plot_cumulative_scores(scores, score_type):
    """
    Plots the cumulative scores of different models based on the given score type.

    Parameters:
    scores (dict): A dictionary containing the scores of different models.
    score_type (str): The type of score to plot. Can be "pedestrian_harm", "passenger_harm", or any other type.

    Returns:
    None
    """

    plt.figure(figsize=(12, 6))
    for model, score_list in scores.items():
        cumulative_scores = [
            sum(score_list[: i + 1]) for i in range(len(score_list))
        ]
        sns.lineplot(
            x=range(1, len(score_list) + 1),
            y=cumulative_scores,
            label=f"{model.capitalize()} Model",
        )
    plt.title(f"{score_type} - Cumulative Scores of Models")
    plt.xlabel("Scenario Number")
    plt.ylabel("Cumulative Score")
    plt.legend()
    plt.savefig(f"plots/{score_type}/cumulative_scoes.png", dpi=300)
    plt.close()
    


def plot_score_distribution(scores, score_type):
    flattened_scores = [
        (model, score) for model, score_list in scores.items() for score in score_list
    ]
    df = pd.DataFrame(flattened_scores, columns=["Model", "Score"])
    cmap = sns.color_palette("mako", as_cmap=True)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Model", y="Score", data=df, cmap=cmap)
    plt.title(f"{score_type} - Score Distribution of Models")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.savefig(f"plots/{score_type}/distribution.png", dpi=300)
    plt.close()

def plot_heatmap(scores, score_type):
    score_matrix = pd.DataFrame(scores)
    cmap = sns.color_palette("mako", as_cmap=True)
    plt.figure(figsize=(12, 8))
    sns.heatmap(score_matrix, annot=True, fmt=".2f", cmap = cmap)
    plt.title(f"{score_type} - Scenario-wise Scores of Models")
    plt.xlabel("Model")
    plt.ylabel("Scenario Number")
    plt.savefig(f"plots/{score_type}/heatmap.png", dpi=300)
    plt.close()

def plot_scores(scores, score_type):
    plt.figure(figsize=(12, 6))
    for model, score_list in scores.items():
        sns.lineplot(
            x=range(1, len(score_list) + 1),
            y=score_list,
            label=f"{model.capitalize()} Model",
        )

    plt.title(f"{score_type} - Model Score Comparison")
    plt.xlabel("Scenario Number")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig(f"plots/{score_type}/scores.png", dpi=300)
    plt.close()

def plot_violin_scores(scores, score_type):
    flattened_scores = [
        (model, score) for model, score_list in scores.items() for score in score_list
    ]
    df = pd.DataFrame(flattened_scores, columns=["Model", "Score"])
    plt.figure(figsize=(10, 6))
    sns.violinplot(x="Model", y="Score", data=df, inner="quart")
    plt.title(f"{score_type} - Distribution of Models Using Violin Plot")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.savefig(f"plots/{score_type}/violin.png", dpi=300)
    plt.close()
def plot_reaction_times(reaction_times):
    plt.figure(figsize=(10, 6))
    sns.distplot(reaction_times, hist=True, kde=False, bins=10, color='blue', hist_kws={'edgecolor':'black'}, kde_kws={'linewidth': 4})
    plt.title('Reaction Times of Manual Driver')
    plt.xlabel('Reaction Time (s)')
    plt.ylabel('Density')
    plt.savefig('plots/reaction_times.png')
    print(reaction_times)
    plt.show()
    plt.close()
def normalize_scores_at_each_index(scores):
    """
    Normalizes the scores such that at each index the choice with the highest score becomes 1 and the lowest becomes 0.

    Parameters:
    scores (iterable of lists): The score arrays for different choices.

    Returns:
    list of lists: Normalized score arrays.
    """
    
    scores = list(scores)  # Ensure scores is a list of lists
    num_choices = len(scores)

    # Check if scores is empty or contains empty lists
    if not scores or any(len(choice_scores) == 0 for choice_scores in scores):
        raise ValueError("Scores should not be empty and should not contain empty lists.")

    # Determine the number of indices (assumes all score lists are of the same length)
    num_indices = len(scores[0])

    # Initialize the list to hold the normalized scores for each choice
    normalized_score_arrays = [[] for _ in range(num_choices)]

    # Normalize the scores at each index
    for index in range(num_indices):
        # Get all scores at this index across different choices
        index_scores = [choice_scores[index] for choice_scores in scores]

        max_score = max(index_scores)
        min_score = min(index_scores)

        # Normalize the score for each choice at this index
        for choice_index, score in enumerate(index_scores):
            normalized_score = 0 if max_score == min_score else (score - min_score) / (max_score - min_score)
            normalized_score_arrays[choice_index].append(normalized_score)

    return normalized_score_arrays
def run_scenarios(client, num_scenarios, winners, choices):
    scores = {choice: [] for choice in choices}
    scores_pedestrian_harm = {choice: [] for choice in choices}
    scores_passenger_harm = {choice: [] for choice in choices}
    reaction_times = []
    turns_per_scenario = {choice: [] for choice in choices}
    scenario_steering_data = {
        num_scenarios: [] for num_scenarios in range(1, num_scenarios + 1)
    }
    for i in range(num_scenarios):
        attributes = generate_scenario_attributes(client)
        scores_per_scenario = []

        for choice in choices:
            if choice != "manual":
                winner = winners.get(choice, None)
                (
                    harm,
                    normalized_pedestrian_harm,
                    normalized_passenger_harm,
                    _,
                    turn,
                    all_steering,
                ) = run_test_scenario(attributes, choice, winner)
                score = 1 - harm
                scores[choice].append(score)
                scores_per_scenario.append(score)
                score_pedestrian_harm = 1 - normalized_pedestrian_harm
                scores_pedestrian_harm[choice].append(score_pedestrian_harm)
                score_passenger_harm = 1 - normalized_passenger_harm
                scores_passenger_harm[choice].append(score_passenger_harm)
                turns_per_scenario[choice].append(turn)
                if choice not in ["left", "right", "straight"]:
                    scenario_steering_data[i + 1] = all_steering
            else:
                (
                    harm,
                    normalized_pedestrian_harm,
                    normalized_passenger_harm,
                    reaction_time,
                    turn,
                    all_steering
                ) = run_test_scenario(attributes, choice, None)
                score = 1 - harm
                scores[choice].append(score)
                score_pedestrian_harm = 1 - normalized_pedestrian_harm
                scores_pedestrian_harm[choice].append(score_pedestrian_harm)
                score_passenger_harm = 1 - normalized_passenger_harm
                scores_passenger_harm[choice].append(score_passenger_harm)
                reaction_times.append(reaction_time)
                turns_per_scenario[choice].append(turn)

                
                normalized_scores = normalize_scores_at_each_index(scores[choice] for choice in choices)
                normalized_pedestrian_scores = normalize_scores_at_each_index(scores_pedestrian_harm[choice] for choice in choices)
                normalized_passenger_scores = normalize_scores_at_each_index(scores_passenger_harm[choice] for choice in choices)
                for i , choice in enumerate(choices):
                    scores[choice] = normalized_scores[i]
                    scores_pedestrian_harm[choice] = normalized_pedestrian_scores[i]
                    scores_passenger_harm[choice] = normalized_passenger_scores[i]
    return (
        scores,
        scores_pedestrian_harm,
        scores_passenger_harm,
        reaction_times,
        turns_per_scenario,
        scenario_steering_data,
    )


def run_scenarios_user_input(client, num_scenarios, winners, choices):
    scores = {choice: [] for choice in choices}
    scores_pedestrian_harm = {choice: [] for choice in choices}
    scores_passenger_harm = {choice: [] for choice in choices}

    for _ in range(num_scenarios):
        attributes = generate_scenario_attributes(client)
        scores_per_scenario = []
        for choice in choices:
            winner = winners.get(choice, None)
            (
                harm,
                normalized_pedestrian_harm,
                normalized_passenger_harm,
            ) = run_test_scenario(attributes, choice, winner)
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
                normalized_score = (
                    0.5  # Handle division by zero if max_score equals min_score
                )
            scores[run_type].append(normalized_score)

    return scores, scores_pedestrian_harm, scores_passenger_harm


if __name__ == "__main__":
    client = carla.Client("localhost", 2000)
    client.set_timeout(15)
    world = client.get_world()
    settings_setter(world)
    sns.set_theme(style="dark")
    sns.set_palette("mako")
    filepaths = ["saved_genomes/genome_9411_fitness_9029.82748334049.pkl"]

    # folder_path = 'saved_genomes'

    # filepaths = []

    # for file in os.listdir(folder_path):

    #     file_path = os.path.join(folder_path, file)
    #     if os.path.isfile(file_path):

    #         filepaths.append(file_path)

    winners = load_winner_nets(filepaths)

    num_scenarios = 100
    choices = list(winners.keys()) + [
        "left",
        "right",
        "straight"#,
#        "manual"
    ] 

    (
        overall_scores,
        pedestrian_scores,
        passenger_scores,
        reaction_times,
        all_turns,
        scenario_steering_data,
    ) = run_scenarios(client, num_scenarios, winners, choices)

    plot_steering_traces(scenario_steering_data)
    # print(reaction_times)
    # plot_reaction_times(reaction_times)

    # plot_average_scores(overall_scores, "Total Score")
    # plot_cumulative_scores(overall_scores, "Total Score")
    # plot_score_distribution(overall_scores, "Total Score")
    # plot_heatmap(overall_scores, "Total Score")
    # plot_scores(overall_scores, "Total Score")
    # plot_violin_scores(overall_scores, "Total Score")

    # plot_average_scores(pedestrian_scores, "Pedestrian Score")
    # plot_cumulative_scores(pedestrian_scores, "Pedestrian Score")
    # plot_score_distribution(pedestrian_scores, "Pedestrian Score")
    # plot_heatmap(pedestrian_scores, "Pedestrian Score")
    # plot_scores(pedestrian_scores, "Pedestrian Score")
    # plot_violin_scores(pedestrian_scores, "Pedestrian Score")

    # plot_average_scores(passenger_scores, "Passenger Score")
    # plot_cumulative_scores(passenger_scores, "Passenger Score")
    # plot_score_distribution(passenger_scores, "Passenger Score")
    # plot_heatmap(passenger_scores, "Passenger Score")
    # plot_scores(passenger_scores, "Passenger Score")
    # plot_violin_scores(passenger_scores, "Passenger Score")
