import pandas as pd
import numpy as np
from datetime import timedelta
from tqdm import tqdm_notebook as tqdm
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

train = pd.read_csv('train.csv').sort_values('impression_time')
test = pd.read_csv('test.csv')
test_save_impession = test.impression_id.values
test = test.sort_values('impression_time')
item_data = pd.read_csv('item_data.csv')
view_log = pd.read_csv('view_log.csv')
view_log['server_time'] = pd.to_datetime(view_log['server_time'])

data = pd.concat([train, test], sort=False).reset_index(drop = True)
ltr = len(train)

data['impression_time'] = pd.to_datetime(data['impression_time'])

def create_(x):
    x = x.values
    return list(zip(x[:, 0], x[:, 1]))

def compute_mean(data, col):
    dict_time_train = data[~data.is_click.isnull()].groupby(col)[['impression_time', 'is_click']].apply(lambda x:
                create_(x)).to_dict()
    diff_time_train = []
    for time, user_id in tqdm(zip(data.impression_time.values, data[col].values)):
        time_s = time - np.timedelta64(0, 'D')
        time_f = time - np.timedelta64(100, 'D')
        if user_id in dict_time_train:
            tmp = [x[1] for x in dict_time_train[user_id] if x[0] < time_s  and x[0] >= time_f]
            if len(tmp) > 0:
                diff_time_train += [(np.sum(tmp) + 20 * 0.04571) / (len(tmp) + 20)]
#                 diff_time_train += [np.mean(tmp)]

            else:
                diff_time_train += [-1]
        else:
            diff_time_train += [-1]
    data['mean_' + col] = diff_time_train
    return data

data = compute_mean(data, 'user_id')


def func(x):
    x = x.values
    if len(x) > 1:
        x = np.diff(x).astype('timedelta64[s]')
        return x.min().astype('int')
    else:
        return -1

tmp_time = data.sort_values(['impression_time']).groupby('user_id').impression_time.apply(lambda x: func(x))
data['tmp_time_min'] = data['user_id'].map(tmp_time)

def compute_in(data, col):
    dict_time_user = data.groupby(col)['impression_time'].apply(np.array).to_dict()
    diff_time_train_0 = []
    diff_time_train_1 = []
    diff_time_train_2 = []
    diff_time_train_3 = []
    for time, user_id in tqdm(zip(data.impression_time.values, data[col].values)):
        time_s = time - np.timedelta64(28, 'D')
        time_f = time + np.timedelta64(1, 'D')
        tmp_0 = [x for x in dict_time_user[user_id] if x< time and x >= time_s]
        tmp_01 = [x for x in dict_time_user[user_id] if x< time]
        tmp_1 = [x for x in dict_time_user[user_id] if x> time and x <= time_f]
        tmp_11 = [x for x in dict_time_user[user_id] if x> time]
        if len(tmp_0) > 0:
            max_time = np.max(tmp_0)
            diff_time_train_0 += [(time - max_time) / np.timedelta64(1, 'D')]
        else:
            diff_time_train_0 += [-1]
        if len(tmp_1) > 0:
            min_time = np.min(tmp_1)
            diff_time_train_1 += [(min_time - time) / np.timedelta64(1, 'D')]
        else:
            diff_time_train_1 += [-1]
        diff_time_train_2 += [len(tmp_01)]
        diff_time_train_3 += [len(tmp_1)]
    data['time_last_' + col] = diff_time_train_0
    data['time_first_' + col] = diff_time_train_1
    data['num_last_' + col] = diff_time_train_2
    data['num_first_' + col] = diff_time_train_3
    return data

data = compute_in(data, 'user_id')

def create_(x):
    x = x.values
    return list(zip(x[:, 0], x[:, 1],  x[:, 2],  x[:, 3]))

tmp = dict(zip(item_data.item_id.values, item_data.item_price.values))

view_log['price'] = view_log['item_id'].map(tmp)

dict_time = view_log.groupby('user_id')[['server_time', 'item_id', 'session_id' , 'price']].apply(lambda x:
                create_(x)).to_dict()

len_items = []
set_items = []
len_uniq = []
price_items = []
for time, user_id in tqdm(zip(data.impression_time, data.user_id)):
    if user_id in dict_time:
        time_start = time - np.timedelta64(100, 'D')
        time_finish = time - np.timedelta64(7, 'D')
        tmp = [x for x in dict_time[user_id] if x[0] >= time_start and x[0] <= time_finish]
        tmp_items = [x[1] for x in tmp]
        tmp_session = [x[2] for x in tmp]
        tmp_price = [x[3] for x in tmp]
        if len(tmp_items) > 0:
            len_items += [len(tmp_items)]
            set_items += [len(set(tmp_session))]
            len_uniq += [len(set(tmp_items))]
            # price_items += [np.nanmean(tmp_price)]
        else:
            len_items += [0]
            set_items += [0]
            len_uniq += [0]
            # price_items += [-1]
    else:
        len_items += [0]
        set_items += [0]
        len_uniq += [0]
        # price_items += [-1]

data['set_items'] = set_items
data['len_uniq'] = len_uniq

data['vc_app'] = data['app_code'].map(data['app_code'].value_counts().to_dict())
data['vc_user_id'] = data['user_id'].map(data['user_id'].value_counts().to_dict())

dict_train = view_log[view_log.server_time <= '2018-12-04'].groupby(['user_id']).size().to_dict()
dict_test = view_log.groupby(['user_id']).size().to_dict()
data['gp_size'] = 0
data.loc[:ltr - 1, 'gp_size'] = data.loc[:ltr - 1, 'user_id'].map(dict_train).values
data.loc[ltr: , 'gp_size'] = data.loc[ltr: , 'user_id'].map(dict_test).values

data['os_version'] = data['os_version'].astype('category').cat.codes
data['diff_items_set'] = data['set_items'] - data['gp_size']

tmp = data.groupby('user_id').app_code.apply(lambda x: len(set(x)))
data['len_uniq_app'] = data['user_id'].map(tmp)
data['diff_uniq_app'] = data['len_uniq_app'] - data['vc_user_id']

data['tmp_1'] = data['app_code'].map(data.groupby(['app_code']).mean_user_id.mean())

data['tmp_2'] = data['app_code'].map(data.groupby(['app_code']).vc_user_id.mean())
data['tmp_3'] = data['app_code'].map(data.groupby(['app_code']).gp_size.mean())

def standart_split(data, n_splits):
    split_list = []
    for i in range(n_splits):
        kf = StratifiedKFold(n_splits=10, shuffle = True, random_state = 228)
        for train_index, test_index in kf.split(data.iloc[:ltr, :], data['is_click'][:ltr]):
            split_list += [(train_index, test_index)]
    return split_list
split_list = standart_split(data, 1)

def lgb_train(data, target, ltr, train_cols, split_list, param, n_e = 10000, cat_col = None, verb_num = None):
    pred = pd.DataFrame()
    pred_val = np.zeros(ltr)
    score = []
    j = 0
    train_pred = pd.DataFrame()

    for i , (train_index, test_index) in enumerate(split_list):
        tr = lgb.Dataset(np.array(data[train_cols])[train_index], np.array(data[target])[train_index])
        te = lgb.Dataset(np.array(data[train_cols])[test_index], np.array(data[target])[test_index], reference=tr)
        tt = lgb.Dataset(np.array(data[train_cols])[ltr:, :])
        evallist = [(tr, 'train'), (te, 'test')]
        bst = lgb.train(param, tr, num_boost_round = n_e,valid_sets = [tr, te], feature_name=train_cols,
                        early_stopping_rounds=150, verbose_eval = verb_num)
        pred[str(i)] =bst.predict(np.array(data[train_cols])[ltr:])
        pred_val[test_index] = bst.predict(np.array(data[train_cols])[test_index])
        score += [metrics.roc_auc_score(np.array(data[target])[test_index], pred_val[test_index])]
        print(i, 'MEAN: ', np.mean(score), 'LAST: ', score[-1])

    train_pred[str(j)] = pred_val
    ans = pd.Series( pred.mean(axis = 1).tolist())
    ans.name = 'lgb'
    return pred, score, train_pred, bst
param_lgb = { 'boosting_type': 'gbdt', 'objective': 'binary', 'metric':'auc',
             'bagging_freq':1, 'subsample':1, 'feature_fraction': 0.7,
              'num_leaves': 8, 'learning_rate': 0.05, 'lambda_l1':5,'max_bin':255}

train_cols = ['os_version', 'tmp_time_min', 'len_uniq_app', 'mean_user_id', 'diff_uniq_app', 'gp_size',
              'tmp_1', 'tmp_3', 'tmp_2',
               'vc_app',  'len_uniq', 'time_last_user_id', 'time_first_user_id', 'num_first_user_id',]
ans_new_2, score_new_2, train_pred_new_2, bst = lgb_train(data, 'is_click', ltr, train_cols,
                       split_list, param_lgb,  verb_num  = 250)

tmp = ans_new_2.copy()
for col in tmp.columns:
    tmp[col] = tmp[col].rank()
tmp = tmp.mean(axis = 1)
tmp  =tmp / tmp.max()

tmp = dict(zip(test.impression_id.values, tmp))
answer1 = pd.DataFrame()
answer1['impression_id'] = test_save_impession
answer1['is_click'] = answer1['impression_id'].map(tmp)
answer1.to_csv('answer.csv', index = None)