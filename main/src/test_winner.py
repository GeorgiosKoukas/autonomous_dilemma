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
    plt.legend()
    plt.show()


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
    plt.show()


def plot_average_scores(scores, score_type):
    average_scores = {
        model: sum(score_list) / len(score_list) for model, score_list in scores.items()
    }

    # Setting a color palette
    sns.set_palette("pastel")

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

    # Setting titles and labels
    if score_type == "pedestrian_harm":
        plt.title("Pedestrian Harm - Average Scores of Models")
        plt.ylabel("Average Score")
    elif score_type == "passenger_harm":
        plt.title("Passenger Harm - Average Scores of Models")
        plt.ylabel("Average Score")
    else:
        score_type = "overall_scores(with Ethical Knob)"
        plt.title("Average Scores of Models")
        plt.ylabel("Average Score")

    plt.xlabel("Model")
    plt.xticks(rotation=45)  # Rotating model names for better visibility
    plt.tight_layout()  # Adjust layout

    # Save plot with a higher resolution
    plt.savefig(f"plots/{score_type}/average_scores.png", dpi=300)
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
            cumulative_scores = [
                sum(score_list[: i + 1]) for i in range(len(score_list))
            ]
            sns.lineplot(
                x=range(1, len(score_list) + 1),
                y=cumulative_scores,
                label=f"{model.capitalize()} Model",
            )
        plt.title("Pedestrian Harm - Cumulative Scores of Models")
        plt.xlabel("Scenario Number")
        plt.ylabel("Cumulative Score")
        plt.legend()
        plt.savefig("plots/pedestrian_harm_scores/cumulative_scores.png")
        plt.show()
    elif score_type == "passenger_harm":
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
        plt.title("Passenger Harm - Cumulative Scores of Models")
        plt.xlabel("Scenario Number")
        plt.ylabel("Cumulative Score")
        plt.legend()
        plt.savefig("plots/passengers_harm_scores/cumulative_scores.png")
        plt.show()
    else:
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
        plt.title("Cumulative Scores of Models")
        plt.xlabel("Scenario Number")
        plt.ylabel("Cumulative Score")
        plt.legend()
        plt.savefig("plots/overall_scores(with Ethical Knob)/cumulative_scores.png")
        plt.show()


def plot_score_distribution(scores, score_type):
    flattened_scores = [
        (model, score) for model, score_list in scores.items() for score in score_list
    ]
    df = pd.DataFrame(flattened_scores, columns=["Model", "Score"])

    if score_type == "pedestrian_harm":
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="Model", y="Score", data=df)
        plt.title("Pedestrian Harm - Score Distribution of Models")
        plt.xlabel("Model")
        plt.ylabel("Score")
        plt.savefig("plots/pedestrian_harm_scores/score_distribution.png")
        plt.show()
    elif score_type == "passenger_harm":
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="Model", y="Score", data=df)
        plt.title("Passenger Harm - Score Distribution of Models")
        plt.xlabel("Model")
        plt.ylabel("Score")
        plt.savefig("plots/passengers_harm_scores/score_distribution.png")
        plt.show()
    else:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="Model", y="Score", data=df)
        plt.title("Score Distribution of Models")
        plt.xlabel("Model")
        plt.ylabel("Score")
        plt.savefig("plots/overall_scores(with Ethical Knob)/score_distribution.png")
        plt.show()


def plot_heatmap(scores, score_type):
    score_matrix = pd.DataFrame(scores)
    if score_type == "pedestrian_harm":
        plt.figure(figsize=(12, 8))
        sns.heatmap(score_matrix, annot=True, fmt=".2f", cmap="viridis")
        plt.title("Pedestrian Harm - Scenario-wise Scores of Models")
        plt.xlabel("Model")
        plt.ylabel("Scenario Number")
        plt.savefig("plots/pedestrian_harm_scores/heatmap.png")
        plt.show()
    elif score_type == "passenger_harm":
        plt.figure(figsize=(12, 8))
        sns.heatmap(score_matrix, annot=True, fmt=".2f", cmap="viridis")
        plt.title("Passenger Harm - Scenario-wise Scores of Models")
        plt.xlabel("Model")
        plt.ylabel("Scenario Number")
        plt.savefig("plots/passengers_harm_scores/heatmap.png")
        plt.show()
    else:
        plt.figure(figsize=(12, 8))
        sns.heatmap(score_matrix, annot=True, fmt=".2f", cmap="viridis")
        plt.title("Scenario-wise Scores of Models")
        plt.xlabel("Model")
        plt.ylabel("Scenario Number")
        plt.savefig("plots/overall_scores(with Ethical Knob)/heatmap.png")
        plt.show()


def plot_scores(scores, score_type):
    sns.set(style="whitegrid")
    if score_type == "pedestrian_harm":
        plt.figure(figsize=(12, 6))
        for model, score_list in scores.items():
            sns.lineplot(
                x=range(1, len(score_list) + 1),
                y=score_list,
                label=f"{model.capitalize()} Model",
            )

        plt.title("Pedestrian Harm - Model Score Comparison")
        plt.xlabel("Scenario Number")
        plt.ylabel("Score")
        plt.legend()
        plt.savefig("plots/pedestrian_harm_scores/scores.png")
        plt.show()
    elif score_type == "passenger_harm":
        plt.figure(figsize=(12, 6))
        for model, score_list in scores.items():
            sns.lineplot(
                x=range(1, len(score_list) + 1),
                y=score_list,
                label=f"{model.capitalize()} Model",
            )

        plt.title("Passenger Harm - Model Score Comparison")
        plt.xlabel("Scenario Number")
        plt.ylabel("Score")
        plt.legend()
        plt.savefig("plots/passengers_harm_scores/scores.png")
        plt.show()
    else:
        plt.figure(figsize=(12, 6))

        for model, score_list in scores.items():
            sns.lineplot(
                x=range(1, len(score_list) + 1),
                y=score_list,
                label=f"{model.capitalize()} Model",
            )

        plt.title("Model Score Comparison")
        plt.xlabel("Scenario Number")
        plt.ylabel("Score")
        plt.legend()
        plt.savefig("plots/overall_scores(with Ethical Knob)/scores.png")
        plt.show()


def plot_violin_scores(scores, score_type):
    flattened_scores = [
        (model, score) for model, score_list in scores.items() for score in score_list
    ]
    df = pd.DataFrame(flattened_scores, columns=["Model", "Score"])
    if score_type == "pedestrian_harm":
        plt.figure(figsize=(10, 6))
        sns.violinplot(x="Model", y="Score", data=df)
        plt.title("Pedestrian Harm - Score Distribution of Models Using Violin Plot")
        plt.xlabel("Model")
        plt.ylabel("Score")
        plt.savefig("plots/pedestrian_harm_scores/violin_scores.png")
        plt.show()
    elif score_type == "passenger_harm":
        plt.figure(figsize=(10, 6))
        sns.violinplot(x="Model", y="Score", data=df)
        plt.title("Passenger Harm - Score Distribution of Models Using Violin Plot")
        plt.xlabel("Model")
        plt.ylabel("Score")
        plt.savefig("plots/passengers_harm_scores/violin_scores.png")
        plt.show()
    else:
        plt.figure(figsize=(10, 6))
        sns.violinplot(x="Model", y="Score", data=df)
        plt.title("Score Distribution of Models Using Violin Plot")
        plt.xlabel("Model")
        plt.ylabel("Score")
        plt.savefig("plots/overall_scores(with Ethical Knob)/violin_scores.png")
        plt.show()


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
                ) = run_test_scenario(attributes, choice, None)
                score = 1 - harm
                scores[choice].append(score)
                scores_per_scenario.append(score)
                score_pedestrian_harm = 1 - normalized_pedestrian_harm
                scores_pedestrian_harm[choice].append(score_pedestrian_harm)
                score_passenger_harm = 1 - normalized_passenger_harm
                scores_passenger_harm[choice].append(score_passenger_harm)
                reaction_times.append(reaction_time)
                turns_per_scenario[choice].append(turn)

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

    filepaths = ["saved_genomes/golden ones/genome_25664_fitness_9209.538459962874.pkl"]

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
        "straight",
    ]  # Add or remove choices as needed
    # choices = ["manual"]

    (
        overall_scores,
        pedestrian_scores,
        passenger_scores,
        reaction_times,
        all_turns,
        scenario_steering_data,
    ) = run_scenarios(client, num_scenarios, winners, choices)
    print(scenario_steering_data)
    plot_steering_traces(scenario_steering_data)
    # plt.figure()
    # sns.histplot(reaction_times, kde=True)
    # plt.title('Reaction Times of Manual Driver')
    # plt.xlabel('Reaction Time (s)')
    # plt.ylabel('Density')
    # plt.savefig('plots/reaction_times.png')
    # plt.show()
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
