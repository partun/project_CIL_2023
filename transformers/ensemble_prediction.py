"""
This script is used to ensemble the predictions of different models.
"""
import pandas as pd
import os
import csv


def ensemble_val(folder_path):
    """
    ensemble_val() is used to ensemble the validation results of different models.
    It takes a folder path as input, and outputs a CSV file containing the ensemble predictions.

    It also prints the validation accuracy of each model, and the ensemble prediction accuracy.
    """

    # Create a list of CSV file names ending with '.csv'
    csv_file_names = [file for file in os.listdir(folder_path) if file.endswith(".csv")]

    df = pd.read_csv(folder_path + csv_file_names[0], sep="\t")

    # Get the length of the 'Prediction' column, which will be used for validation
    validation_length = len(df["Prediction"].tolist())

    ensemble_scores = [0] * validation_length

    # Loop through all the CSV files in the folder
    for csv_file_name in csv_file_names:
        csv_file_path = folder_path + csv_file_name

        df = pd.read_csv(csv_file_path, sep="\t")

        predictions = df["Prediction"].tolist()
        labels = df["Label"].tolist()

        # Calculate the number of correct predictions ('true_count') by comparing 'Prediction' with 'Label'
        true_count = 0
        for prediction, label in zip(predictions, labels):
            if prediction == label:
                true_count += 1

        print(csv_file_name, "Validation accuracy:", true_count / validation_length)

        # Update the 'ensemble_scores' list by adding the current predictions to the existing scores
        ensemble_scores = [a + b for a, b in zip(ensemble_scores, predictions)]

    ids = [i + 1 for i in range(validation_length)]
    ensemble_predictions = [1 if score > 0 else -1 for score in ensemble_scores]

    data = list(zip(ids, ensemble_predictions))
    header = ["Id", "Prediction"]

    # Write Prediction data into a new CSV file
    output_file_name = "ensemble_val_predictions.csv"
    with open(output_file_name, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(data)

    # Calculate the ensemble prediction accuracy
    true_count = 0
    print(len(ensemble_predictions), len(labels), validation_length)

    for i in range(validation_length):
        if ensemble_predictions[i] == labels[i]:
            true_count += 1

    print(f"CSV file '{output_file_name}' has been generated.")
    print("Ensemble Prediction", true_count / validation_length)


def ensamble_test(folder_path):
    """
    ensemble_test() is used to ensemble the submission csv files of different models.
    """

    csv_file_names = [file for file in os.listdir(folder_path) if file.endswith(".csv")]
    df = pd.read_csv(folder_path + csv_file_names[0], sep=",")

    # Get the length of the 'Prediction' column, which will be used for validation
    validation_length = len(df["Prediction"].tolist())

    ensemble_scores = [0] * validation_length

    # Loop through all the CSV files in the folder
    for csv_file_name in csv_file_names:
        csv_file_path = folder_path + csv_file_name

        df = pd.read_csv(csv_file_path, sep=",")

        predictions = df["Prediction"].tolist()

        # Update the 'ensemble_scores' list by adding the current predictions to the existing scores
        ensemble_scores = [a + b for a, b in zip(ensemble_scores, predictions)]

    ids = [i + 1 for i in range(validation_length)]
    ensemble_predictions = [1 if score > 0 else -1 for score in ensemble_scores]

    data = list(zip(ids, ensemble_predictions))
    header = ["Id", "Prediction"]

    # Write Prediction data into a new CSV file
    output_file_name = "ensemble_predictions.csv"
    with open(output_file_name, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(data)

    print(f"CSV file '{output_file_name}' has been generated.")


if __name__ == "__main__":
    # ensamble of validatio predictions modify the path to the folder containing the validation predictions
    # ensemble_val("val_results/")

    # ensamble of test predictions modify the path to the folder containing the test predictions
    ensamble_test("test_results/")
