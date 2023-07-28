import pandas as pd
import os
import csv

folder_path = 'results/'
file_names = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

scores = [0] * 10000
for file_name in file_names:
    file_path = folder_path + file_name
# Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    prediction = df['Prediction'].tolist()
    scores = [a + b for a, b in zip(scores, prediction)]
    

ids = [i+1 for i in range(10000)]
predictions = [1 if score > 0 else -1 for score in scores]
# print(id, predictions)
data = list(zip(ids, predictions))
header = ["Id", "Prediction"]
print(scores)
file_name = "ensemble_predictions.csv"

with open(file_name, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    writer.writerows(data)

print(f"CSV file '{file_name}' has been generated.")
