import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import skew, kurtosis


# Define the dataset paths
eyes_closed_paths = [
    # "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A1-01/A1_01_EC.csv",
    # "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A1-02/A1_02_EC.csv",
    # "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A1-03/A1_03_EC.csv",
    # "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A1-04/A1_04_EC.csv",
    # "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A1-05/A1_05_EC.csv",
    # "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A1-06/A1_06_EC.csv",
    # "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A1-07/A1_07_EC.csv",

    # "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A2-01/A2_01_EC.csv",
    # "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A2-02/A2_02_EC.csv",
    # "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A2-03/A2_03_EC.csv",
    # "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A2-04/A2_04_EC.csv",
    # "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A2-05/A2_05_EC.csv",
    # "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A2-06/A2_06_EC.csv",
    # "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A2-07/A2_07_EC.csv",

    # "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/B_01/B_01_EC.csv",
    # "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/B_02/B_02_EC.csv",
    # "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/B_03/B_03_EC.csv",
    # "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/B_04/B_04_EC.csv",
    # "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/B_05/B_05_EC.csv",
    # "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/B_06/B_06_EC.csv",
    # "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/B_07/B_07_EC.csv",
    # "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/B_08/B_08_EC.csv",
    # "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/B_09/B_09_EC.csv",

    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/C_1/C_1_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/C_2/C_2_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/C_3/C_3_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/C_4/C_4_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/C_5/C_5_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/C_6/C_6_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/C_7/C_7_EC.csv",
    "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/C_8/C_8_EC.csv"
]

    # "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/B_10/B_10_EC.csv",

eyes_open_paths = [
    # "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A1-01/A1_01_EO.csv",
    # "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A1-02/A1_02_EO.csv",
    # "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A1-03/A1_03_EO.csv",
    # "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A1-04/A1_04_EO.csv",
    # "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A1-05/A1_05_EO.csv",
    # "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A1-06/A1_06_EO.csv",
    # "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A1-07/A1_07_EO.csv",

    #  "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A2-01/A2_01_EO.csv",
    # "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A2-02/A2_02_EO.csv",
    # "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A2-03/A2_03_EO.csv",
    # "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A2-04/A2_04_EO.csv",
    # "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A2-05/A2_05_EO.csv",
    # "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A2-06/A2_06_EO.csv",
    # "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A2-07/A2_07_EO.csv",

    # "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/B_01/B_01_EO.csv",
    # "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/B_02/B_02_EO.csv",
    # "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/B_03/B_03_EO.csv",
    # "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/B_04/B_04_EO.csv",
    # "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/B_05/B_05_EO.csv",
    # "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/B_06/B_06_EO.csv",
    #  "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/B_07/B_07_EO.csv",
    # "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/B_08/B_08_EO.csv",
    # "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/B_09/B_09_EO.csv",

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
    # (17, 24, 16, 20.67, 32, "A1-01", "Imran"),
    # (12, 32, 19, 13.00, 45, "A1-02", "Ijat"),
    # (22, 27, 21, 3.67, 30, "A1-03", "Malik"),
    # (18, 27, 14, 15.67, 29, "A1-04", "Faris"),
    # (25, 16, 21, 16.00, 36, "A1-05", "Anif"),
    # (14, 22, 16, 11.33, 24, "A1-06", "Taib"),
    # (17, 27, 18, 9.00, 29, "A1-07", "Syazwan"),

    # (20, 22, 16, 19.33, 32, "A2-01", "MUHAMMAD 'ILYAS AMIERRULLAH BIN AB KARIM"),
    # (15, 19, 18, 13.33, 23, "A2-02", "hazim imran bin abd aziz"),
    # (11, 19, 12, 20, 19, "A2-03", "Harraz Nasrullah bin Suhairul Afzan"),
    # (18, 28, 19, 14.33, 35, "A2-04", "Adam Ashraf bin Azlan"),
    # (19, 25, 19, 10.33, 39, "A2-05", "hakimnazry@gmail.com"),
    # (11, 8, 16, 19.67, 36, "A2-06", "Nisa Nabilah binti Azaddin"),
    # (16, 21, 20, 11.66666667,32,"A2-07","Nurinhany Mysara binti Noor Haslan"),

    # (18, 24, 21, 22.33, 33, "B-01", "NUR AMIRA SYAFIQAH BINTI MOHD FAUZI"),
    # (22, 21, 18, 16.33, 41, "B-02", "NUR ALYSSA FITRI BINTI MOHAMAD FAUZI"),
    # (21, 19, 16, 16.67, 37, "B-03", "Amir Azim Bin Mohd Kamaruzaman"),
    # (20, 32, 27, 9.00, 28, "B-04", "Nur Adila Binti Muhammad Zahid"),
    # (19, 23, 23, 16.33, 33, "B-05", "NURUL ADLINA BINTI ROSLAN"),
    # (24, 31, 32, 11.67, 39, "B-06", "NURAIN AWATIF BINTI ISMAIL"),
    # # (18, 22, 3, 17.33, 27, "B-07", "NUR ANNISA BINTI MOHAMAD MAHAYUDIN"),
    # (19, 26, 24, 22.67, 34, "B-08", "Puteri Nur Sabrina Binti Mohd Azlee"),
    # (20, 24, 17, 16.00, 29, "B-09", "Noor Afiqah Binti Normadi"),

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
from mne.channels import make_dig_montage
import os

def plot_eeg_topomap_muse_from_csv(csv_path, save_path_animation=None, save_directory=None, show_names=False, start_time=0.05, end_time=1, step_size=0.1):
    """
    Plot EEG topomap for Muse data from a CSV file using MNE.

    Args:
        csv_path (str): Path to the CSV file containing Muse EEG data.
        save_path_animation (str): Name of the file to save the animation (optional).
        save_directory (str): Directory to save the file (optional).
        show_names (bool): Whether to show channel names on the topomap.
        start_time (float): Starting time for the topomap plot (in seconds).
        end_time (float): Ending time for the topomap plot (in seconds).
        step_size (float): Step size for time increments in the topomap plot (in seconds).
    """

    # Load data from the CSV
    data = pd.read_csv(csv_path)  # Use csv_path here
    print(f"Loaded data with shape: {data.shape}")

     # Extract relevant channels (Muse channels: AF7, AF8, TP9, TP10)
    muse_channels = ['AF7', 'AF8', 'TP9', 'TP10']
    if not all(ch in data.columns for ch in muse_channels):
        raise ValueError(f"The dataset must contain the following channels: {muse_channels}")

    eeg_data = data[muse_channels].values.T  # Transpose to match shape (n_channels, n_times)

    # Define Muse channel positions
    muse_positions = {
        'AF7': [-0.05, 0.085, 0],
        'AF8': [0.05, 0.085, 0],
        'TP9': [-0.08, -0.04, 0],
        'TP10': [0.08, -0.04, 0]
    }

    # Add a small random offset to channel positions to avoid Qhull error
    for ch in muse_positions:
        muse_positions[ch][0] += np.random.normal(0, 0.0001)  # Add small random value to x
        muse_positions[ch][1] += np.random.normal(0, 0.0001)  # Add small random value to y

    montage = make_dig_montage(ch_pos=muse_positions, coord_frame='head')

    # Calculate sampling rate from timestamps
    timestamps = data['timestamps']
    time_diffs = np.diff(timestamps)
    average_interval = np.mean(time_diffs)
    sfreq = 1 / average_interval  # Sampling rate

    # Create MNE Info and Evoked objects
    info = mne.create_info(ch_names=muse_channels, sfreq=sfreq, ch_types='eeg')
    evoked = mne.EvokedArray(eeg_data, info)

    # Set montage
    evoked.set_montage(montage)

    # Define time points
    times = np.arange(start_time, end_time, step_size)

    # Plot the topomap
    evoked.plot_topomap(times, ch_type='eeg', time_unit='s', ncols=5, nrows=2, show_names=show_names)

    # Save the animation
    if save_path_animation:
        if save_directory:
            os.makedirs(save_directory, exist_ok=True)
            save_path_animation = os.path.join(save_directory, save_path_animation)

        from matplotlib.animation import FuncAnimation
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        def update(frame):
            ax.clear()
            evoked.plot_topomap([times[frame]], ch_type='eeg', time_unit='s', axes=ax, colorbar=False, show=False)

        anim = FuncAnimation(fig, update, frames=len(times), interval=200)
        anim.save(save_path_animation, writer='imagemagick')
        print(f"Animation saved to: {save_path_animation}")
        plt.close(fig)



# Define file_path with the path to your CSV file
# file_path = "https://raw.githubusercontent.com/6661647a77616e/Muse-Dataset/main/dataset/A1-01/A1_01_EO.csv"  # Replace with your actual path

save_directory = "animations"
# save_file_name = "eeg_topomap_animation.gif"
# plot_eeg_topomap_muse_from_csv(file_path, save_path_animation=save_file_name, save_directory=save_directory, show_names=True)

for subject_id, subject_info in dataset_dict.items():
    save_file_name = f'{subject_id}.gif'
    plot_eeg_topomap_muse_from_csv(dataset_dict[subject_id]["eyes_open_path"], save_path_animation=save_file_name, save_directory=save_directory, show_names=True)