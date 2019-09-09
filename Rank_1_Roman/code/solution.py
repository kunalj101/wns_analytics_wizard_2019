import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import numpy as np
from tqdm import tqdm
from scipy import stats



train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

item_data = pd.read_csv('item_data.csv')
view_data = pd.read_csv('view_log.csv')



view_data['server_time'] = pd.to_datetime(view_data['server_time'])
view_data = view_data.sort_values('server_time').reset_index(drop = True)



train['impression_time'] = pd.to_datetime(train['impression_time'])
test['impression_time'] = pd.to_datetime(test['impression_time'])

train = train.sort_values('impression_time').reset_index(drop = True)
test = test.sort_values('impression_time').reset_index(drop = True)

ltr = len(train)

data = pd.concat([train, test], axis = 0).reset_index(drop = True)



data['diff_time_mean'] = data['user_id'].map(
    data.groupby('user_id')['impression_time'].apply(lambda x: np.nanmean(x.diff() / np.timedelta64(1, 's'))).to_dict())
data['diff_time_max'] = data['user_id'].map(
    data.groupby('user_id')['impression_time'].apply(lambda x: np.nanmax(x.diff() / np.timedelta64(1, 's'))).to_dict())
data['diff_time_min'] = data['user_id'].map(
    data.groupby('user_id')['impression_time'].apply(lambda x: np.nanmin(x.diff() / np.timedelta64(1, 's'))).to_dict())



data['os_version'] = data['os_version'].astype('category').cat.codes



data['cnt_unique_app'] = data['user_id'].map(data.groupby('user_id')['app_code'].apply(lambda x: x.unique().size).to_dict())



data['hour'] = data['impression_time'].dt.hour
data['minute'] = data['impression_time'].dt.minute



for col in ['app_code', 'user_id']:
    data['vc_' + col] = data[col].map(data[col].value_counts().to_dict())



view_data = view_data.merge(item_data, how = 'left', on = 'item_id')




data['app_code_cnt_unique_user'] = data['app_code'].map(
    data.groupby('app_code')['user_id'].apply(lambda x: x.unique().size).to_dict())




value_list = []
value_list_last = []
for user_id, imp_time in tqdm(zip(data.user_id.values, data.impression_time.values)):
    cur_data = data[(data.impression_time < imp_time) & (data.user_id == user_id)]
    value_list += [cur_data['is_click'].mean()]
    if cur_data.shape[0] > 0:
        value_list_last += [(imp_time - cur_data['impression_time'].values[-1]) /  np.timedelta64(1, 's')]
    else:
        value_list_last += [np.nan]
data['value_mean_user_id'] = value_list
data['diff_time_user_id_last'] = value_list_last



value_list = []
value_list_last = []
for app_code, imp_time in tqdm(zip(data.app_code.values, data.impression_time.values)):
    cur_data = data[(data.impression_time < imp_time) & (data.app_code == app_code)]
    value_list += [cur_data['is_click'].mean()]
    if cur_data.shape[0] > 0:
        value_list_last += [(imp_time - cur_data['impression_time'].values[-1]) /  np.timedelta64(1, 's')]
    else:
        value_list_last += [np.nan]

data['value_mean_app_code'] = value_list
data['diff_time_app_code_last'] = value_list_last



value_list_last = []
for app_code, imp_time in tqdm(zip(data.app_code.values, data.impression_time.values)):
    cur_data = data[(data.impression_time > imp_time) & (data.app_code == app_code)]
    if cur_data.shape[0] > 0:
        value_list_last += [(cur_data['impression_time'].values[0] - imp_time) /  np.timedelta64(1, 's')]
    else:
        value_list_last += [np.nan]
data['diff_time_app_code_next'] = value_list_last



value_list_last = []
for user_id, imp_time in tqdm(zip(data.user_id.values, data.impression_time.values)):
    cur_data = data[(data.impression_time > imp_time) & (data.user_id == user_id)]
    if cur_data.shape[0] > 0:
        value_list_last += [(cur_data['impression_time'].values[0] - imp_time) /  np.timedelta64(1, 's')]
    else:
        value_list_last += [np.nan]
data['diff_time_user_id_next'] = value_list_last



value_list_cnt = []
value_list_diff_mean = []
value_list_diff_max = []
value_list_diff_min = []

value_list_unique_session_id = []
value_list_unique_item_id = []
value_list_unique_category_1 = []
value_list_unique_category_2 = []
value_list_unique_category_3 = []
value_list_unique_product_type = []

value_list_mode_item_id = []
value_list_mode_category_1 = []
value_list_mode_category_2 = []
value_list_mode_category_3 = []
value_list_mode_product_type = []

for user_id, imp_time in tqdm(zip(data.user_id.values, data.impression_time.values)):
    cur_data = view_data[(view_data.server_time < imp_time) & (view_data.user_id == user_id)]
    value_list_cnt += [cur_data.shape[0]]
    
    if cur_data.shape[0] == 0:
        value_list_unique_session_id += [0]
        value_list_unique_item_id += [0]
        value_list_unique_category_1 += [0]
        value_list_unique_category_2 += [0]
        value_list_unique_category_3 += [0]
        value_list_unique_product_type += [0]

        value_list_mode_item_id += [-1]
        value_list_mode_category_1 += [-1]
        value_list_mode_category_2 += [-1]
        value_list_mode_category_3 += [-1]
        value_list_mode_product_type += [-1]
    
    else:
    
        value_list_unique_session_id += [cur_data['session_id'].unique().size]
        value_list_unique_item_id += [cur_data['item_id'].unique().size]
        value_list_unique_category_1 += [cur_data['category_1'].unique().size]
        value_list_unique_category_2 += [cur_data['category_2'].unique().size]
        value_list_unique_category_3 += [cur_data['category_3'].unique().size]
        value_list_unique_product_type += [cur_data['product_type'].unique().size]
        
        if len(cur_data['item_id'].value_counts()) > 0:
            value_list_mode_item_id += [cur_data['item_id'].value_counts().index[0]]
        else:
            value_list_mode_item_id += [-1]
            
        if len(cur_data['category_1'].value_counts()) > 0:
            value_list_mode_category_1 += [cur_data['category_1'].value_counts().index[0]]
        else:
            value_list_mode_category_1 += [-1]
        
        if len(cur_data['category_2'].value_counts()) > 0:
            value_list_mode_category_2 += [cur_data['category_2'].value_counts().index[0]]
        else:
            value_list_mode_category_2 += [-1]
        
        
        if len(cur_data['category_3'].value_counts()) > 0:
            value_list_mode_category_3 += [cur_data['category_3'].value_counts().index[0]]
        else:
            value_list_mode_category_3 += [-1]
        
        if len(cur_data['product_type'].value_counts()) > 0:
            value_list_mode_product_type += [cur_data['product_type'].value_counts().index[0]]
        else:
            value_list_mode_product_type += [-1]
    
    value_list_diff_mean += [cur_data['server_time'].diff().mean()]
    value_list_diff_max += [cur_data['server_time'].diff().max()]
    value_list_diff_min += [cur_data['server_time'].diff().min()]
    
data['value_cnt_view_user_id'] = value_list_cnt

data['user_id_unique_session_id'] = value_list_unique_session_id
data['user_id_unique_item_id'] = value_list_unique_item_id
data['user_id_unique_category_1'] = value_list_unique_category_1
data['user_id_unique_category_2'] = value_list_unique_category_2
data['user_id_unique_category_3'] = value_list_unique_category_3
data['user_id_unique_product_type'] = value_list_unique_product_type

data['user_id_mode_item_id'] = value_list_mode_item_id
data['user_id_mode_category_1'] = value_list_mode_category_1
data['user_id_mode_category_2'] = value_list_mode_category_2
data['user_id_mode_category_3'] = value_list_mode_category_3
data['user_id_mode_product_type'] = value_list_mode_product_type

data['value_diff_time_view_user_id_mean'] = value_list_diff_mean
data['value_diff_time_view_user_id_max'] = value_list_diff_max
data['value_diff_time_view_user_id_min'] = value_list_diff_min



data['value_diff_time_view_user_id_mean'] = data['value_diff_time_view_user_id_mean'] / np.timedelta64(1, 's')
data['value_diff_time_view_user_id_max'] = data['value_diff_time_view_user_id_max'] / np.timedelta64(1, 's')
data['value_diff_time_view_user_id_min'] = data['value_diff_time_view_user_id_min'] / np.timedelta64(1, 's')



cols = ['user_id_mode_item_id', 'user_id_mode_category_1',
       'user_id_mode_category_2', 'user_id_mode_category_3',
       'user_id_mode_product_type']

for col in cols:
    data['vc_' + col] = data[col].map(data[col].value_counts().to_dict())



data['gg_1_diff'] = data['app_code'].map(data.groupby('app_code')['user_id'].apply(lambda x: len(x) - x.unique().size).to_dict())
data['gg_1_ratio'] = data['app_code'].map(data.groupby('app_code')['user_id'].apply(lambda x: x.unique().size / len(x)).to_dict())

data['gg_2_diff'] = data['user_id'].map(data.groupby('user_id')['app_code'].apply(lambda x: len(x) - x.unique().size).to_dict())
data['gg_2_ratio'] = data['user_id'].map(data.groupby('user_id')['app_code'].apply(lambda x: x.unique().size / len(x)).to_dict())



train_cols = ['is_4G',
       'os_version',  'diff_time_mean', 'diff_time_max',
       'diff_time_min', 'cnt_unique_app', 'hour', 'minute',
       'vc_app_code', 'vc_user_id', 'app_code_cnt_unique_user',
       'value_mean_user_id', 'diff_time_user_id_last', 'value_mean_app_code',
       'diff_time_app_code_last', 'diff_time_app_code_next',
       'diff_time_user_id_next', 'value_cnt_view_user_id',
       'user_id_unique_session_id', 'user_id_unique_item_id',
       'user_id_unique_category_1', 'user_id_unique_category_2',
       'user_id_unique_category_3', 'user_id_unique_product_type',
      'value_diff_time_view_user_id_mean',
       'value_diff_time_view_user_id_max', 'value_diff_time_view_user_id_min', 'vc_user_id_mode_item_id', 'vc_user_id_mode_category_1',
       'vc_user_id_mode_category_2', 'vc_user_id_mode_category_3',
       'vc_user_id_mode_product_type', 'gg_1_diff', 'gg_1_ratio', 'gg_2_diff',
       'gg_2_ratio']



param_lgb = {
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'boost': 'gbdt',
    'feature_fraction': 0.8,
    'learning_rate': 0.01,
    'metric':'auc',
    'num_leaves': 31,
    'num_threads': 8,
    'objective': 'binary',
#     'lambda_l1':1
#     'lambda_l2':30
}



kf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 228)


pred = pd.DataFrame()
score = []

for i , (train_index, test_index) in enumerate(kf.split(data.loc[:ltr-1, :], data.loc[:ltr-1, 'is_click'])):
    tr = lgb.Dataset(np.array(data[train_cols])[train_index], np.array(data['is_click'])[train_index])
    te = lgb.Dataset(np.array(data[train_cols])[test_index], np.array(data['is_click'])[test_index],
                     reference=tr)
    bst = lgb.train(param_lgb, tr, num_boost_round=10000, 
            valid_sets=te, early_stopping_rounds=int(5 / param_lgb['learning_rate']), verbose_eval = 100)

    pred[str(i)] = bst.predict(np.array(data[train_cols])[ltr:, :])



ans = pd.DataFrame()
ans['impression_id'] = data.loc[ltr:, 'impression_id'].values
ans['is_click'] = (pred.rank().mean(axis = 1) / len(pred)).values



ans.to_csv('answer.csv', index = None)

