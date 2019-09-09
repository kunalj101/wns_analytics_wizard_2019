import pandas as pd
import numpy as np
from scipy.stats import rankdata
from sklearn.preprocessing import MinMaxScaler

preds = pd.DataFrame()
subA = pd.read_csv("sub_1.csv")
subB = pd.read_csv("sub_2.csv")

subA = subA.merge(subB, on='impression_id', how='left')

preds['impression_id'] = subA.impression_id
preds['PredsSubA'] = subA.is_click_x
preds['PredsSubB'] = subA.is_click_y
preds['RanksSubA'] = rankdata(subA.is_click_x)
preds['RanksSubB'] = rankdata(subA.is_click_y)
preds['RankAverage'] = preds[['RanksSubA', 'RanksSubB']].mean(1)
preds['is_click'] = MinMaxScaler().fit_transform(preds['RankAverage'].values.reshape(-1, 1))
preds[['impression_id', 'is_click']].to_csv("final_submission.csv", index=False)