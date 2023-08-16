from pathlib import Path
from utils import calculate_correlation, root_mean_squared_error, med_absolute_error, save_data, load_data
import numpy as np
import pandas as pd

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import RootMeanSquaredError, MeanSquaredError, MeanAbsoluteError
from nn_arch import ann136

NAME = 'ann136'
ROOT = Path(__file__).cwd()
PATH_TO_DATA = ROOT.joinpath('data')
ROOT.joinpath(f'{NAME}_models').mkdir(parents=True, exist_ok=True)
PATH_TO_SAVE_MODEL = ROOT.joinpath(f'{NAME}_models//model')
ROOT.joinpath(f'{NAME}_models//metrics').mkdir(parents=True, exist_ok=True)
PATH_TO_SAVE_DATA = ROOT.joinpath(f'{NAME}_models//metrics')
RESULTS_FILE = ROOT.joinpath(f'{NAME}_models//results.txt')

# Loss functions
rmse = RootMeanSquaredError()
mse = MeanSquaredError()
mae = MeanAbsoluteError()

LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
history_dict = dict()
corr_dict = dict()
rmse_dict = dict()
medae_dict = dict()


def create_one_hot(train, test):
    f_lst = ['ecg_id', 'superclasses', 'heart_axis']
    df_train = pd.DataFrame(train, columns=f_lst)
    df_test = pd.DataFrame(test, columns=f_lst)
    n = df_train.shape[0]

    features = pd.concat([df_train, df_test])
    heart_axis = pd.get_dummies(features, columns=['heart_axis'], drop_first=True)

    ha_and_sc = heart_axis.loc[:, 'heart_axis_1':'heart_axis_8']
    one_hot_train = ha_and_sc.iloc[:n]
    one_hot_test = ha_and_sc.iloc[n:]
    return one_hot_train, one_hot_test


def main():
    for i, lead in enumerate(LEADS):
        # Targets
        y = ecg_train[:, :, i]
        y_test = ecg_test[:, :, i]
        # ECG lead inversion
        # if lead == 'aVR':
        #     y = -1 * y
        #     y_test = -1 * y_test

        # Create model
        model = ann136()
        model.compile(optimizer=Adam(0.00001), loss='mse', metrics=[rmse, mae])
        # Early learning stop function
        stop_fit = EarlyStopping(monitor='val_loss', min_delta=0.00005, patience=5, restore_best_weights=True)
        # Training
        history = model.fit([X, one_hot_train], y, batch_size=4096, shuffle=True, epochs=150, validation_split=0.2,
                            callbacks=[stop_fit])
        # Saving training data
        history_dict[lead] = history.history
        model.save(f'{PATH_TO_SAVE_MODEL}_{lead}')

        # Prediction
        y_predict = model.predict([X_test, one_hot_test])
        y_pred = np.squeeze(y_predict, axis=2)
        del model

        # Calculating and saving metrics
        corr, mean_corr, std_corr = calculate_correlation(y_test, y_pred)
        corr_dict[lead] = corr
        rmse_, mean_rmse, rmse_std = root_mean_squared_error(y_test, y_pred)
        rmse_dict[lead] = rmse_
        medae, mean_medae, medae_std = med_absolute_error(y_test, y_pred)
        medae_dict[lead] = medae

        # Writing metrics to a file
        with open(RESULTS_FILE, 'a', encoding='UTF8', newline='') as f:
            f.write(f"{lead}: RMSE = {mean_rmse} \u00B1 {rmse_std}, MedAE = {mean_medae} \u00B1 {medae_std},"
                    f" CORR = {mean_corr} \u00B1 {std_corr}\n")

    # Writing data to disc
    save_data(history_dict, PATH_TO_SAVE_DATA.joinpath('history_dict.pkl'))
    save_data(corr_dict, PATH_TO_SAVE_DATA.joinpath('corr_dict.pkl'))
    save_data(rmse_dict, PATH_TO_SAVE_DATA.joinpath('rmse_dict.pkl'))
    save_data(medae_dict, PATH_TO_SAVE_DATA.joinpath('medae_dict.pkl'))


# Loading data for network training
ecg_train = load_data(PATH_TO_DATA, 'ecg_train_clean.pkl')
ecg_test = load_data(PATH_TO_DATA, 'ecg_test_clean.pkl')
# Heart axis
features_train = load_data(PATH_TO_DATA, 'features_train_clean.pkl')
features_test = load_data(PATH_TO_DATA, 'features_test_clean.pkl')
one_hot_train, one_hot_test = create_one_hot(features_train, features_test)

# I Lead is input data
X = ecg_train[:, :, 0]
X_test = ecg_test[:, :, 0]


if __name__ == '__main__':
    main()
