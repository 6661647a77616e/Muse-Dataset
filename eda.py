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
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/A1-04/A1_04_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/A1-05/A1_05_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/A1-06/A1_06_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/A1-07/A1_07_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/C_1/C_1_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/C_2/C_2_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/C_3/C_3_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/C_4/C_4_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/C_5/C_5_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/C_6/C_6_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/C_7/C_7_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/C_8/C_8_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/C_9/C_9_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/C_10/C_10_EC.csv",
]

eyes_open_paths = [
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A1-01/A1_01_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A1-02/A1_02_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A1-03/A1_03_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A1-04/A1_04_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A1-05/A1_05_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A1-06/A1_06_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A1-07/A1_07_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/C_1/C_1_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/C_2/C_2_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/C_3/C_3_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/C_4/C_4_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/C_5/C_5_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/C_6/C_6_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/C_7/C_7_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/C_8/C_8_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/C_9/C_9_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/C_10/C_10_EO.csv",
]
data = [
    (17, 24, 16, 20.67, 32, "A1-01", "Imran"),
    (12, 32, 19, 13.00, 45, "A1-02", "Ijat"),
    (22, 27, 21, 3.67, 30, "A1-03", "Malik"),
    (18, 27, 14, 15.67, 29, "A1-04", "Faris"),
    (25, 16, 21, 16.00, 36, "A1-05", "Anif"),
    (14, 22, 16, 11.33, 24, "A1-06", "Taib"),
    (17, 27, 18, 9.00, 29, "A1-07", "Syazwan"),
    (17, 26, 19, 34, "C-01", "Dahlia Binti Husin"),
    (13, 21, 28, 26, 38, "C-02", "Nur Fathnin Ilhami Bt Mohamad Helmi"),
    (20, 13, 13, 21.67, 27, "C-03", "Noor Azzatul Sofea Binti Yayah"),
    (20, 29, 22, 13, 31, "C-04", "Raihanatuzzahra Binti Azmi"),
    (20, 26, 9, 14, 33, "C-05", "Hayani Nazurah Binti Hasram"),
    (19, 18, 7, 23, 39, "C-06", "Nur Syakirah Huda Binti Razali"),
    (10, 26, 19, 17.67, 27, "C-07", "Qhuratul Nissa Binti Mior Idris"),
    (24, 17, 15, 23.33, 34, "C-08", "Ainul Mardhiah Binti Mohammed Rafiq")
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