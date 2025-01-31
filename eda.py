import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import skew, kurtosis


# Define the dataset paths
eyes_closed_paths = [
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/A1-01/A1_01_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/A1-02/A1_02_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/A1-03/A1_03_EC.csv",
    # "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/A1-04/A1_04_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/A1-05/A1_05_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/A1-06/A1_06_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/A1-07/A1_07_EC.csv"
]

eyes_open_paths = [
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/A1-01/A1_01_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/A1-02/A1_02_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/A1-03/A1_03_EO.csv",
    # "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/A1-04/A1_04_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/A1-05/A1_05_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/A1-06/A1_06_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/A1-07/A1_07_EO.csv"
]

# Define the metadata manually
data = [
    (17, 24, 16, 20.67, 32, "A1-01", "Imran"),
    (12, 32, 19, 13.00, 45, "A1-02", "Ijat"),
    (22, 27, 21, 3.67, 30, "A1-03", "Malik"),
    # (18, 27, 14, 15.67, 29, "A1-04", "Faris"),
    (25, 16, 21, 16.00, 36, "A1-05", "Anif"),
    (14, 22, 16, 11.33, 24, "A1-06", "Taib"),
    (17, 27, 18, 9.00, 29, "A1-07", "Syazwan")
]

# Create a dictionary mapping each subject ID to their corresponding details
dataset_dict = {}

for i, (ext, agr, con, new, ope, id_name, name) in enumerate(data):
    dataset_dict[id_name] = {
        "ext": ext,
        "agr": agr,
        "con": con,
        "new": new,
        "ope": ope,
        "id_name": id_name,
        "name": name,
        "eyes_open_path": eyes_open_paths[i],
        "eyes_closed_path": eyes_closed_paths[i]
    }

# Convert the dictionary to a Pandas DataFrame (Optional)
df = pd.DataFrame.from_dict(dataset_dict, orient="index")

# Display the DataFrame
print(df.head())
print("shape",df.shape)
print(df.info())

# Select each column from the DataFrame
ext_column = df['ext']
agr_column = df['agr']
con_column = df['con']
new_column = df['new']
ope_column = df['ope']
id_name_column = df['id_name']
name_column = df['name']
eyes_open_path_column = df['eyes_open_path']
eyes_closed_path_column = df['eyes_closed_path']

# Display the selected columns
print("Extraversion Column:\n", ext_column)
print("Agreeableness Column:\n", agr_column)
print("Conscientiousness Column:\n", con_column)
print("Neuroticism Column:\n", new_column)
print("Openness Column:\n", ope_column)
print("ID Name Column:\n", id_name_column)
print("Name Column:\n", name_column)
print("Eyes Open Path Column:\n", eyes_open_path_column)
print("Eyes Closed Path Column:\n", eyes_closed_path_column)