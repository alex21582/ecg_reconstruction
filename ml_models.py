from pathlib import Path
from utils import calculate_correlation, root_mean_squared_error, med_absolute_error, save_data, load_data
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearnex import patch_sklearn

patch_sklearn(['KNeighborsRegressor', 'Ridge'])

ROOT = Path(__file__).cwd()
PATH_TO_DATA = ROOT.joinpath('data')
PATH_TO_SAVE_MODEL, PATH_TO_SAVE_DATA, RESULTS_FILE = Path(), Path(), Path()
LEADS = ['II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def create_dirs(name):
    global PATH_TO_SAVE_MODEL, PATH_TO_SAVE_DATA, RESULTS_FILE

    ROOT.joinpath(f'{name}_models').mkdir(parents=True, exist_ok=True)
    PATH_TO_SAVE_MODEL = ROOT.joinpath(f'{name}_models//model')
    ROOT.joinpath(f'{name}_models//metrics').mkdir(parents=True, exist_ok=True)
    PATH_TO_SAVE_DATA = ROOT.joinpath(f'{name}_models//metrics')
    RESULTS_FILE = ROOT.joinpath(f'{name}_models//results.txt')


def create_ml_models(ml_model=None):
    corr_dict = dict()
    rmse_dict = dict()
    medae_dict = dict()

    X = ecg_train[:, :, 0]
    X_test = ecg_test[:, :, 0]

    for i, lead in enumerate(LEADS):
        y = ecg_train[:, :, i + 1]
        y_test = ecg_test[:, :, i + 1]

        # if i == 3:
        #     y = -1 * y
        #     y_test = -1 * y_test
        if ml_model is None:
            y_pred = X_test
        else:
            model = ml_model.fit(X, y)
            save_data(model, f'{PATH_TO_SAVE_MODEL}_{lead}.pkl')
            y_pred = model.predict(X_test)
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

    save_data(corr_dict, PATH_TO_SAVE_DATA.joinpath('corr_dict.pkl'))
    save_data(rmse_dict, PATH_TO_SAVE_DATA.joinpath('rmse_dict.pkl'))
    save_data(medae_dict, PATH_TO_SAVE_DATA.joinpath('medae_dict.pkl'))


if __name__ == '__main__':
    ecg_train = load_data(PATH_TO_DATA, 'ecg_train_clean.pkl')
    ecg_test = load_data(PATH_TO_DATA, 'ecg_test_clean.pkl')

    create_dirs('knnr')
    knnr = KNeighborsRegressor(n_neighbors=15, weights='distance', n_jobs=-1)
    create_ml_models(knnr)

    create_dirs('ridge')
    ridge = Ridge(alpha=1)
    create_ml_models(ridge)

    create_dirs('base')
    create_ml_models()
