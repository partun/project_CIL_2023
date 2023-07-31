import pandas as pd
import os
import csv

folder_path = 'val_results/'

# Create a list of CSV file names ending with '.csv'
csv_file_names = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

df = pd.read_csv(folder_path + csv_file_names[0], sep='\t')

# Get the length of the 'Prediction' column, which will be used for validation
validation_length = len(df['Prediction'].tolist())

ensemble_scores = [0] * validation_length

# Loop through all the CSV files in the folder
for csv_file_name in csv_file_names:
    csv_file_path = folder_path + csv_file_name
    print(csv_file_path)

    df = pd.read_csv(csv_file_path, sep='\t')

    predictions = df['Prediction'].tolist()
    labels = df['Label'].tolist()

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
output_file_name = "ensemble_predictions.csv"
with open(output_file_name, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    writer.writerows(data)

# Calculate the ensemble prediction accuracy
true_count = 0
for i in range(validation_length):
    if ensemble_predictions[i] == labels[i]:
        true_count += 1

print(f"CSV file '{output_file_name}' has been generated.")
print("Ensemble Prediction", true_count / validation_length)
