import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import skew, kurtosis



# Define the dataset paths
eyes_closed_paths = [
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A1-01/A1_01_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A1-02/A1_02_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A1-03/A1_03_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A1-04/A1_04_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A1-05/A1_05_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A1-06/A1_06_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A1-07/A1_07_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A2-01/A2_01_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A2-02/A2_02_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A2-03/A2_03_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A2-04/A2_04_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A2-05/A2_05_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A2-06/A2_06_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A2-06/A2_07_EC.csv",
    # "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/B_1/B_1_EC.csv"
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/B_02/B_02_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/B_03/B_03_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/B_04/B_04_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/B_05/B_05_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/B_06/B_06_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/B_07/B_07_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/B_08/B_08_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/B_09/B_09_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/B_10/B_10_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/C_1/C_1_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/C_2/C_2_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/C_3/C_3_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/C_4/C_4_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/C_5/C_5_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/C_6/C_6_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/C_7/C_7_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/C_8/C_8_EC.csv"
]

eyes_open_paths = [
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A1-01/A1_01_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A1-02/A1_02_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A1-03/A1_03_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A1-04/A1_04_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A1-05/A1_05_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A1-06/A1_06_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A1-07/A1_07_EO.csv",
     "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A2-01/A2_01_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A2-02/A2_02_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A2-03/A2_03_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A2-04/A2_04_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A2-05/A2_05_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A2-06/A2_06_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A2-06/A2_07_EO.csv",
    # "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/B_1/B_1_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/B_02/B_02_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/B_03/B_03_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/B_04/B_04_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/B_05/B_05_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/B_06/B_06_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/B_07/B_07_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/B_08/B_08_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/B_09/B_09_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/B_10/B_10_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/C_1/C_1_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/C_2/C_2_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/C_3/C_3_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/C_4/C_4_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/C_5/C_5_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/C_6/C_6_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/C_7/C_7_EO.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/C_8/C_8_EO.csv"
]

data = [
    (17, 24, 16, 20.67, 32, "A1-01", "Imran"),
    (12, 32, 19, 13.00, 45, "A1-02", "Ijat"),
    (22, 27, 21, 3.67, 30, "A1-03", "Malik"),
    (18, 27, 14, 15.67, 29, "A1-04", "Faris"),
    (25, 16, 21, 16.00, 36, "A1-05", "Anif"),
    (14, 22, 16, 11.33, 24, "A1-06", "Taib"),
    (17, 27, 18, 9.00, 29, "A1-07", "Syazwan"),
    (20, 22, 16, 19.33, 32, "A2-01", "MUHAMMAD 'ILYAS AMIERRULLAH BIN AB KARIM"),
    (15, 19, 18, 13.33, 23, "A2-02", "hazim imran bin abd aziz"),
    (11, 19, 12, 20, 19, "A2-03", "Harraz Nasrullah bin Suhairul Afzan"),
    (18, 28, 19, 14.33, 35, "A2-04", "Adam Ashraf bin Azlan"),
    (19, 25, 19, 10.33, 39, "A2-05", "hakimnazry@gmail.com"),
    (11, 8, 16, 19.67, 36, "A2-06", "Nisa Nabilah binti Azaddin"),
    (16, 21, 20, 11.66666667,32,"A2-07","Nurinhany Mysara binti Noor Haslan"),
    (19, 26, 24, 22.67, 34, "B-2", "Puteri Nur Sabrina Binti Mohd Azlee"),
    (19, 23, 23, 16.33, 33, "B-3", "NURUL ADLINA BINTI ROSLAN"),
    (24, 31, 32, 11.67, 39, "B-4", "NURAIN AWATIF BINTI ISMAIL"),
    (20, 24, 17, 16.00, 29, "B-5", "Noor Afiqah Binti Normadi"),
    (20, 32, 27, 9.00, 28, "B-6", "Nur Adila Binti Muhammad Zahid"),
    (18, 22, 3, 17.33, 27, "B-7", "NUR ANNISA BINTI MOHAMAD MAHAYUDIN"),
    (22, 21, 18, 16.33, 41, "B-8", "NUR ALYSSA FITRI BINTI MOHAMAD FAUZI"),
    (18, 24, 21, 22.33, 33, "B-9", "NUR AMIRA SYAFIQAH BINTI MOHD FAUZI"),
    (21, 19, 16, 16.67, 37, "B-10", "Amir Azim Bin Mohd Kamaruzaman"),
    (16, 17, 26, 19, 34, "C-01", "Dahlia Binti Husin"),
    (13, 21, 28, 26, 38, "C-02", "Nur Fathnin Ilhami Bt Mohamad Helmi"),
    (20, 13, 13, 21.67, 27, "C-03", "Noor Azzatul Sofea Binti Yayah"),
    (20, 29, 22, 13, 31, "C-04", "Raihanatuzzahra Binti Azmi"),
    (20, 26, 9, 14, 33, "C-05", "Hayani Nazurah Binti Hasram"),
    (19, 18, 7, 23, 39, "C-06", "Nur Syakirah Huda Binti Razali"),
    (10, 26, 19, 17.67, 27, "C-07", "Qhuratul Nissa Binti Mior Idris"),
    (24, 17, 15, 23.33, 34, "C-08", "Ainul Mardhiah Binti Mohammed Rafiq"),
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
    print(df[channels].values.T)
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
    print(">>>>>>>>>>>>>>",subject_info["id_name"],subject_info["name"].upper())
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