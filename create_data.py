from pathlib import Path
import pandas as pd
import numpy as np
from utils import get_random_ecg, get_ecg, get_minmax, save_data

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(__file__).cwd()
# Path to the directory with the data file
DB_CSV_PATH = ROOT.joinpath('csv_files')
#
f_lst = ['ecg_id', 'superclasses', 'heart_axis']
# ECG lead numbers and names
leads = list(range(12))
cols = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


# The function deletes ECG recordings labelled
# electrodes_problems, pacemaker, burst_noise, static_noise
def get_df(data_frame):
    data_frame['heart_axis'] = data_frame['heart_axis'].fillna('NOT')
    data_frame['superclasses'] = data_frame['superclasses'].fillna('NOT')
    data_frame = data_frame.fillna(0)
    data_frame = data_frame[data_frame['hr'] != 0]
    df_clean = data_frame[(data_frame['electrodes_problems'] == 0) & (data_frame['pacemaker'] == 0)
                          & (data_frame['burst_noise'] == 0) & (data_frame['static_noise'] == 0)]
    df_out = df_clean[['ecg_id', 'superclasses', 'heart_axis', 'filename_lr']]
    return df_out


# One Hot feature coding
def create_labels(data_frame, column_name):
    label_encoder = LabelEncoder()
    data_frame[column_name] = label_encoder.fit_transform(data_frame[colunm_name])
    with open(f'{column_name}_labeling.txt', 'w') as f:
        for i, label in enumerate(label_encoder.classes_):
            f.write(f'{i}: {label}\n')
    return data_frame


# The function creates and saves a boxplot image to analyse ECG fragments by amplitude
def save_img_minmax(data_frame, name='train'):
    Path(ROOT.joinpath('images')).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4, 5))
    sns.boxplot(data=data_frame, orient='hor')
    plt.xlabel('Amplitude, mV')
    plt.ylabel('Leads')
    plt.savefig(Path('images').joinpath(f'{name}_minmax.jpg'), bbox_inches='tight', dpi=600)


if __name__ == '__main__':
    df = pd.read_csv(DB_CSV_PATH.joinpath('data_base.csv'))
    df_cut = get_df(df)
    df_cut = create_labels(df_cut, 'heart_axis')
    df_cut = create_labels(df_cut, 'superclasses')

    # Splitting the training sample into training and test samples
    train, test = train_test_split(df_cut, shuffle=True, test_size=0.2, random_state=10)

    # Augmentation procedure
    ecg_aug, features_aug = get_random_ecg(train, leads, 128, 10, f_lst)
    # Splitting ECG signals into fragments
    ecg_train, features_train = get_ecg(train, leads, 128, f_lst)
    ecg_train = np.vstack((ecg_train, ecg_aug))
    features_train = np.vstack((features_train, features_aug))

    ecg_test, features_test = get_ecg(test, leads, 128, f_lst)

    minmax_train = get_minmax(ecg_train)
    minmax_test = get_minmax(ecg_test)

    df_minmax_train = pd.DataFrame(minmax_train, columns=cols)
    df_minmax_test = pd.DataFrame(minmax_test, columns=cols)

    save_img_minmax(df_minmax_train)
    save_img_minmax(df_minmax_test, name='test')

    # Cleaning data from amplitude outliers
    for i in cols:
        df_minmax_train = df_minmax_train[df_minmax_train[i] < df_minmax_train[i].quantile(q=0.99)]
        df_minmax_test = df_minmax_test[df_minmax_test[i] < df_minmax_test[i].quantile(q=0.99)]

    indx_train = df_minmax_train.index
    indx_test = df_minmax_test.index

    save_img_minmax(df_minmax_train, name='train_clean')
    save_img_minmax(df_minmax_test, name='test_clean')

    # Create cleaned data
    ecg_train_clean = ecg_train[indx_train, :, :]
    ecg_test_clean = ecg_test[indx_test, :, :]
    features_train_clean = features_train[indx_train]
    features_test_clean = features_test[indx_test]

    # Save data
    Path(ROOT.joinpath('data')).mkdir(parents=True, exist_ok=True)
    save_data(ecg_train_clean, Path('data').joinpath('ecg_train_clean.pkl'))
    save_data(ecg_test_clean, Path('data').joinpath('ecg_test_clean.pkl'))
    save_data(features_train_clean, Path('data').joinpath('features_train_clean.pkl'))
    save_data(features_test_clean, Path('data').joinpath('features_test_clean.pkl'))


