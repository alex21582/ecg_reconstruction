from pathlib import Path
import pandas as pd
from operator import itemgetter
import neurokit2 as nk
import numpy as np
import wfdb
from tqdm import tqdm
import joblib
from scipy.stats import pearsonr
from config import PTDB_XL_PATH


PATH_TO_DB_FILES = Path(PTDB_XL_PATH).resolve()


def dict_pars_scp(scp_codes_dict):
    scp_codes_list = []
    for item in scp_codes_dict.items():
        scp_codes_list.append(item)
    sorted_scp_codes_list = sorted(scp_codes_list, key=itemgetter(1))
    return sorted_scp_codes_list[-1][0]


def dict_pars_prob(scp_codes_dict):
    scp_codes_list = []
    for item in scp_codes_dict.items():
        scp_codes_list.append(item)
    sorted_scp_codes_list = sorted(scp_codes_list, key=itemgetter(1))
    return sorted_scp_codes_list[-1][1]


def find_ecg_peaks(sig, sampling_rate):
    info_ecg = nk.ecg_findpeaks(sig, sampling_rate)
    ecg_peaks = info_ecg["ECG_R_Peaks"]
    if len(ecg_peaks):
        return ecg_peaks
    else:
        return None


def control_data(data, d=4, out=False):
    q = [1 / 4, 3 / 4]
    if data is None:
        return None
    quants = np.quantile(data, q)
    IQR = quants[1] - quants[0]
    min_data = quants[0] - 1.5 * IQR
    max_data = quants[1] + 1.5 * IQR
    data_proof = data[(data >= min_data) & (data <= max_data)]
    if out:
        return data_proof, np.around(np.mean(data_proof), decimals=d)
    else:
        return np.around(np.mean(data_proof), decimals=d)


def calculate_heart_r(input_files, lead=0):
    ecg_signal, fields = wfdb.rdsamp(PATH_TO_DB_FILES.joinpath(input_files), channels=[lead])
    fs = fields['fs']
    ecg = nk.ecg_clean(ecg_signal, sampling_rate=fs, method='neurokit')
    ecg_peaks = find_ecg_peaks(ecg, sampling_rate=fs)
    if ecg_peaks is None:
        return None
    rate = nk.ecg_rate(peaks=ecg_peaks, sampling_rate=fs, desired_length=len(ecg))
    mean_rate = control_data(rate, d=0, out=False)
    return mean_rate


def get_random_ecg(input_file: pd.DataFrame, lead: list, seg_length: int, n_segments: int, f_lst: list) -> (np.array, np.array):
    # Число элементов в базе
    n_objects = input_file.shape[0]
    # Длина сигнала
    objects_length = 1000

    # Размерность выходного массифа
    axis0 = n_objects * n_segments
    axis1 = seg_length  # 512
    axis2 = len(lead)  # max = 12
    # Пустые массивы
    ecg_array = np.zeros((axis0, axis1, axis2))
    ecg_filt = np.zeros((axis2, objects_length))
    futures_array = np.zeros((axis0, len(f_lst)), dtype='int')

    j = 0
    for row in tqdm(input_file.iterrows(), total=n_objects):
        # Читаем с диска сигнал
        file = row[1]['filename_lr']
        ecg_signal, fields = wfdb.rdsamp(PATH_TO_DB_FILES.joinpath(file), channels=lead)

        # Фильтруем все 12 отведений
        for l in range(axis2):
            ecg_filt[l, :] = nk.signal_filter(ecg_signal[:, l], sampling_rate=fields['fs'], lowcut=0.7, highcut=40,
                                              order=5)

        # Выбираем случайные фрагменты и сохраняем в 3D массив
        for _ in range(n_segments):
            rand_ind = np.random.randint(0, objects_length - seg_length - 1)

            for l in range(axis2):
                ecg_array[j, :, l] = ecg_filt[l, rand_ind: rand_ind + seg_length]
            # Сохраняем признаки
            futures_array[j, :] = row[1][f_lst]
            j += 1
    return ecg_array, futures_array


def get_ecg(input_file: pd.DataFrame, lead: list, seg_length: int, f_lst: list) -> (np.array, np.array):
    n_objects = input_file.shape[0]
    objects_length = 1000
    n_segments = int(objects_length / seg_length)
    axis0 = n_objects * n_segments
    axis1 = seg_length
    axis2 = len(lead)
    ecg_array = np.zeros((axis0, axis1, axis2))
    ecg_filt = np.zeros((axis2, objects_length))
    futures_array = np.zeros((axis0, len(f_lst)), dtype='int')
    j = 0
    for row in tqdm(input_file.iterrows(), total=n_objects):
        file = row[1]['filename_lr']
        ecg_signal, fields = wfdb.rdsamp(PATH_TO_DB_FILES.joinpath(file), channels=lead)
        for l in range(axis2):
            ecg_filt[l, :] = nk.signal_filter(ecg_signal[:, l], sampling_rate=fields['fs'], lowcut=0.7, highcut=40,
                                              order=5)
        for i in range(n_segments):
            for l in range(axis2):
                ecg_array[j, :, l] = ecg_filt[l, i*seg_length: (i + 1)*seg_length]
            futures_array[j, :] = row[1][f_lst]
            j += 1
    return ecg_array, futures_array


def get_minmax(array):
    n = array.shape[0]
    m = 12
    out = np.zeros((n, m))
    for i in tqdm(range(n)):
        for j in range(m):
            out[i, j] = np.max(array[i, :, j]) - np.min(array[i, :, j])
    return out


def save_data(data_file, path_to_file):
    with open(path_to_file, 'wb') as f:
        joblib.dump(data_file, f, compress='zlib')


def load_data(path_to_file, file):
    path = path_to_file.joinpath(file)
    with open(path, 'rb') as f:
        return joblib.load(f)


def calculate_correlation(y_true, y_pred):
    correlations = []
    for i in range(len(y_true)):
        correlation, _ = pearsonr(y_true[i], y_pred[i])
        correlations.append(correlation)
    mean_correlation = np.mean(correlations)
    return correlations, np.round(mean_correlation, decimals=4), np.round(np.std(correlations), decimals=4)


def root_mean_squared_error(y_true, y_pred):
    output_errors = np.average((y_true - y_pred) ** 2, axis=0)
    output_errors = np.sqrt(output_errors)
    return output_errors, np.round(np.average(output_errors), decimals=4), np.round(np.std(output_errors), decimals=4)


def med_absolute_error(y_true, y_pred):
    output_errors = np.median(np.abs(y_pred - y_true), axis=0)
    return output_errors, np.round(np.average(output_errors), decimals=4), np.round(np.std(output_errors), decimals=4)


if __name__ == '__main__':
    pass
