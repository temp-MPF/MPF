import numpy as np
import pandas as pd
import numpy as np
from copy import deepcopy
import torch



def metric_fn_txnx1(pred: pd.Series, label: pd.Series, date_col="datetime", dropna=False):
    # label_transform = label.groupby("datetime").rank(pct=True)
    # label_transform -= 0.5
    # label_transform *= 3.46
    # diff = pred.to_numpy() - label_transform.to_numpy()
    # mse = np.nanmean(diff ** 2)
    # # mae = np.nanmean(diff.abs())
    # mae = np.nanmean(np.abs(diff))

    df = pd.DataFrame({"pred": pred, "label": label})
    ic_ts = df.groupby(date_col).apply(lambda df: df["pred"].corr(df["label"]))
    ric_ts = df.groupby(date_col).apply(lambda df: df["pred"].corr(df["label"], method="spearman"))

    ic, icir, ric, ricir = ic_ts.mean(), ic_ts.mean() / (ic_ts.std() + 1e-8), \
                           ric_ts.mean(), ric_ts.mean() / (ric_ts.std() + 1e-8)

    precision = {}
    recall = {}
    temp = df.groupby(level='datetime').apply(lambda x: x.sort_values(by='pred', ascending=False))
    if len(temp.index[0]) > 2:
        temp = temp.reset_index(level =0).drop('datetime', axis = 1)
    
    for k in [ 5, 10,  30, 50]:
        precision["p_{}".format(int(k))] = temp.groupby(level='datetime').apply(lambda x:(x.label[:k]>0).sum()/k).mean()
        # recall['r_{}'.format(int(k))] = temp.groupby(level='datetime').apply(lambda x:(x.label[:k]>0).sum()/(x.label>0).sum()).mean()

    final_reuslt =  {
        # 'mse': mse,
        # 'mae': mae,
        'ic': ic,
        'icir': icir,
        'ric': ric,
        'ricir': ricir,
    }
    final_reuslt.update(precision)
    final_reuslt.update(recall)
    return final_reuslt
