import ast
from pathlib import Path
import pandas as pd
from utils import dict_pars_scp, dict_pars_prob, calculate_heart_r

# Root directory
ROOT = Path(__file__).cwd()
# PTB-XL database directory
DB_CSV_PATH = ROOT.joinpath('csv_files')


# Creating a dictionary with the superclass feature
def create_superclass():
    sclass_dict = dict()
    superclasses = list(agg_df['diagnostic_class'].unique())
    for superclass in superclasses:
        for i in list(agg_df[agg_df['diagnostic_class'] == superclass].index):
            sclass_dict[i] = superclass
    return sclass_dict


# Adding additional features to the database
def fit_db(data_base):
    data_base.insert(loc=2, column='scp', value=0)
    data_base['scp'] = data_base['scp_codes'].map(dict_pars_scp)
    data_base.insert(loc=3, column='scp_prob', value=0)
    data_base['scp_prob'] = data_base['scp_codes'].map(dict_pars_prob)
    data_base.insert(loc=2, column='superclasses', value=df['scp'])
    data_base['superclasses'] = data_base['superclasses'].map(superclass_dict)
    data_base['hr'] = data_base['filename_hr'].map(calculate_heart_r)
    return data_base


if __name__ == '__main__':
    df = pd.read_csv(DB_CSV_PATH.joinpath('ptbxl_database.csv'), index_col='ecg_id')
    df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))
    agg_df = pd.read_csv(DB_CSV_PATH.joinpath('scp_statements.csv'), index_col=0)
    superclass_dict = create_superclass()
    df = fit_db(df)
    df.to_csv(DB_CSV_PATH.joinpath("data_base.csv"))
