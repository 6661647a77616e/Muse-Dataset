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
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/A1-07/A1_07_EC.csv"
]

eyes_open_paths = [
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/A1-01/A1_01_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/A1-02/A1_02_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/A1-03/A1_03_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/A1-04/A1_04_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/A1-05/A1_05_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/A1-06/A1_06_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/A1-07/A1_07_EO.csv"
]

# Define the metadata manually
data = [
    (17, 24, 16, 20.67, 32, "A1-01", "Imran"),
    (12, 32, 19, 13.00, 45, "A1-02", "Ijat"),
    (22, 27, 21, 3.67, 30, "A1-03", "Malik"),
    (18, 27, 14, 15.67, 29, "A1-04", "Faris"),
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
# print(df)


def load_and_combine_data(eyes_opened_path, eyes_closed_path):
    df_eyes_opened = pd.read_csv(eyes_opened_path)
    df_eyes_closed = pd.read_csv(eyes_closed_path)
    
    df_eyes_opened['condition'] = 'eyes opened'
    df_eyes_closed['condition'] = 'eyes closed'
    
    combined_df = pd.concat([df_eyes_opened, df_eyes_closed], ignore_index=True)
    return combined_df

def classify_trait(score):
    return 'Low' if score <= 20 else 'High'

def add_trait_classifications(df, traits):
    for trait, score in traits.items():
        df[trait] = classify_trait(score)
    return df

def preprocess_eeg_data(df, channels, sfreq):
    eeg_data = df[channels].values.T * 1e-6  # Convert µV to Volts
    info = mne.create_info(ch_names=channels, sfreq=sfreq, ch_types=['eeg'] * len(channels))
    raw = mne.io.RawArray(eeg_data, info)
    raw_filtered = raw.copy().filter(l_freq=1.0, h_freq=30.0)
    
    ica = mne.preprocessing.ICA(n_components=len(channels), random_state=42, max_iter=200)
    ica.fit(raw_filtered)
    raw_cleaned = raw_filtered.copy()
    ica.apply(raw_cleaned)
    
    return raw_cleaned.get_data()

def extract_features(segmented_data, sfreq, combined_df):
    all_features = []
    
    def time_domain_features(epoch):
        return {
            "mean": np.mean(epoch),
            "std_dev": np.std(epoch),
            "skewness": skew(epoch),
            "kurtosis": kurtosis(epoch)
        }
    
    def frequency_domain_features(epoch, fs):
        f, psd = welch(epoch, fs, nperseg=len(epoch))
        return {
            "total_power": np.sum(psd),
            "delta_power": np.sum(psd[(f >= 0.5) & (f < 4)]),
            "theta_power": np.sum(psd[(f >= 4) & (f < 8)]),
            "alpha_power": np.sum(psd[(f >= 8) & (f < 13)]),
            "beta_power": np.sum(psd[(f >= 13) & (f < 30)])
        }
    
    for i, epoch in enumerate(segmented_data):
        time_features = time_domain_features(epoch)
        freq_features = frequency_domain_features(epoch, sfreq)
        combined_features = {**time_features, **freq_features}
        
        for trait in ["Extraversion", "Agreeableness", "Openness", "Conscientiousness", "Neuroticism", "condition"]:
            combined_features[trait.lower()] = combined_df[trait].iloc[i]
        
        all_features.append(combined_features)
    
    return pd.DataFrame(all_features)

def main(df, subject_id):
    subject_info = df.loc[subject_id]
    eyes_opened_path = subject_info["eyes_open_path"]
    eyes_closed_path = subject_info["eyes_closed_path"]
    
    combined_df = load_and_combine_data(eyes_opened_path, eyes_closed_path)
    
    traits = {
        "Extraversion": subject_info["ext"],
        "Agreeableness": subject_info["agr"],
        "Conscientiousness": subject_info["con"],
        "Neuroticism": subject_info["new"],
        "Openness": subject_info["ope"]
    }
    
    combined_df = add_trait_classifications(combined_df, traits)
    
    channels = ['TP9', 'AF7', 'AF8', 'TP10']
    sfreq = 256
    cleaned_data = preprocess_eeg_data(combined_df, channels, sfreq)[0]
    
    epoch_length = 1
    samples_per_epoch = int(epoch_length * sfreq)
    num_epochs = len(cleaned_data) // samples_per_epoch
    segmented_data = cleaned_data[:num_epochs * samples_per_epoch].reshape(num_epochs, samples_per_epoch)
    
    features_df = extract_features(segmented_data, sfreq, combined_df)
    return features_df


if __name__ == "__main__":
    final_df = pd.DataFrame()
    for subject_id in df.index:
        result = main(df, subject_id)
        result['subject_id'] = subject_id
        result['name'] = df.loc[subject_id]['name']
        final_df = pd.concat([final_df, result], ignore_index=True)
    
    final_df.to_csv('final_results.csv', index=False)
    print("Results saved to final_results.csv")