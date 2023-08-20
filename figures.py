import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(__file__).cwd()
DB_CSV_PATH = ROOT.joinpath('csv_files')
df = pd.read_csv(DB_CSV_PATH.joinpath('data_base.csv'), index_col='ecg_id')


# Function to create a pie chart for disease superclasses
def superclasess_pi(data_frame):
    labels = ['Normal ECG (NORM)',
              'Myocardial Infarction (MI)',
              'Other (NOT)',
              'ST/T Change (STTC)',
              'Hypertrophy (HYP)',
              'Conduction Disturbance (CD)']

    data_frame['superclasses'] = data_frame['superclasses'].fillna('NOT')
    # superclasses = data_frame['superclasses'].unique()
    data = data_frame['superclasses'].value_counts(normalize=True)

    Path(ROOT.joinpath('images')).mkdir(parents=True, exist_ok=True)
    colors = sns.color_palette('colorblind')
    explode = [0.01] * len(data)
    plt.figure(figsize=(6, 6))
    plt.pie(data, explode=explode, colors=colors, autopct='%.0f%%', textprops={'fontsize': 12})
    plt.legend(labels, loc='upper left', fontsize=9.3)
    plt.savefig(ROOT.joinpath('images//superclasses_pi.jpg'), bbox_inches='tight', dpi=600)


# Function to create a pie chart for the electrical axis of the heart
def heart_axis_pi(data_frame):
    labels = ['Unknown (NOT)',
              'Normal axis (MID)',
              'Left axis deviation (LAD)',
              'Abnormal left axis deviation (ALAD)',
              'Other']

    data_frame['heart_axis'] = data_frame['heart_axis'].fillna('Unknown')
    data = data_frame['heart_axis'].value_counts(normalize=True)
    other_dict = dict()

    for i in range(len(data)):
        if data[i] <= 0.05:
            other_dict[data.index[i]] = 'Other'
    data_frame = df.replace({'heart_axis': other_dict})
    data = data_frame['heart_axis'].value_counts(normalize=True)

    Path(ROOT.joinpath('images')).mkdir(parents=True, exist_ok=True)
    colors = sns.color_palette('colorblind')
    explode = [0.01] * len(data)
    plt.figure(figsize=(6, 6))
    plt.pie(data, explode=explode, colors=colors, autopct='%.0f%%', textprops={'fontsize': 12})
    plt.legend(labels, loc='upper left', fontsize=9.3)
    plt.savefig(ROOT.joinpath('images//heart_axis_pi.jpg'), bbox_inches='tight', dpi=600)


if __name__ == '__main__':
    heart_axis_pi(df)
    superclasess_pi(df)
