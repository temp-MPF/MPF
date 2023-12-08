import argparse
import json
import sys, os

sys.path.append("../")


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="")
parser.add_argument("--instruments", type=str, default="csi100")
parser.add_argument("--data_dir", type=str, default="./results_csv")
parser.add_argument("--out_dir", type=str, default="./results_csv")
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--repeat_times", type=int, default=1)
# for data config
parser.add_argument("--time_step", type=int, default=20)
parser.add_argument("--n_features", type=int, default=165)
# parser.add_argument("--diff_len_num", type=int, default=3)
parser.add_argument("--num_states", type=int, default=5)

# for train
parser.add_argument("--start_repeat", type=int, default=0)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--early_stop", type=int, default=20)
parser.add_argument("--pretrained_cp_path", type=str, default="")

# for model
parser.add_argument("--dropout", type=float, default=0.2)
# parser.add_argument("--rnn_num_heads", type=int, default=2)
parser.add_argument("--tra_hidden_size", type=int, default=32)
parser.add_argument("--rnn_hidden_size", type=int, default=64)
# parser.add_argument('--rnn_num_layers',type= int,default=2)
# parser.add_argument('--rnn_hidden_size',type= int,default=64)
# parser.add_argument('--rnn_dropout',type= float,default=0.0)
# parser.add_argument('--lr_schedule_step',type= str,default=json.dumps([5,10,15,20]))
parser.add_argument("--patterns", type=str, default=json.dumps([":"]))
parser.add_argument("--model_type", type=str, default="LSTM")
parser.add_argument(
    "--length_after_resample", type=str, default=json.dumps([80, 60, 60, 60])
)
parser.add_argument("--not_use_attn", action="store_false", help="default True")


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpu_id)


from pathlib import Path
import qlib

import pandas as pd
from qlib.data.dataset import TSDatasetH, DatasetH
from qlib.contrib.data.dataset import MTSDatasetH

# from tra_dataset import MTSDatasetH
from qlib.config import REG_CN
import time
import torch
from qlib.contrib.data.handler import Alpha158, Alpha360
from qlib.data.dataset import TSDatasetH
import numpy as np
import datetime


from models.pytorch_mpf_single import MPFSingleModel as Model
from utils_src.utils import metric_fn_txnx1

# pretrained_cp_path = os.path.join(ar)
provider_uri = args.data_dir  # os.path.join(args.data_dir,args.provider_uri)
qlib.init(provider_uri=provider_uri, region=REG_CN)  # 

torch.backends.cudnn.benchmark = False

hander_config = {
    "start_time": datetime.date(2008, 1, 1),
    "end_time": datetime.date(2020, 8, 1),
    "fit_start_time": datetime.date(2008, 1, 1),
    "fit_end_time": datetime.date(2014, 12, 31),
    "instruments": args.instruments,
    "infer_processors": [
        {
            "class": "RobustZScoreNorm",
            "kwargs": {
                "fields_group": "feature",
                "clip_outlier": True,
                "fit_start_time": datetime.date(2008, 1, 1),
                "fit_end_time": datetime.date(2014, 12, 31),
            },
        },
        {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
    ],
    "learn_processors": [
        {"class": "DropnaLabel"},
        {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
    ],
    "label": ["Ref($close, -2) / Ref($close, -1) - 1"],
    "time_step": args.time_step,
}
h = Alpha360(**hander_config)
dataset = MTSDatasetH(
    **{
        "handler": h,
        "segments": {
            "train": ("2008-01-01", "2014-12-31"),
            "valid": ("2015-01-01", "2016-12-31"),
            "test": ("2017-01-01", "2020-08-01"),
        },
        "seq_len": 60,
        "horizon": 2,
        "input_size": 6,
        "num_states": args.num_states,
        "batch_size": 1024,
        "memory_mode": "sample",
        "drop_last": True,
    }
)

n_features = args.n_features

final_result = []
out_dir = args.out_dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
if not os.path.exists(os.path.join(out_dir, "csv")):
    os.makedirs(os.path.join(out_dir, "csv"))
if not os.path.exists(os.path.join(out_dir, "checkpoint")):
    os.makedirs(os.path.join(out_dir, "checkpoint"))
if len(args.pretrained_cp_path) != 0:
    pretrained_cp_path_all = "T"
else:
    pretrained_cp_path_all = ""
base_config_list = [
    args.model,
    args.instruments,
    args.time_step,
    args.n_features,
    args.lr,
    args.batch_size,
    args.start_repeat,
    args.repeat_times,
    args.num_states,
    args.model_type,
    str(args.not_use_attn),
]

result_path = os.path.join(
    out_dir, "csv", "{}.csv".format("_".join([str(v) for v in base_config_list]))
)


for repeat_idx in range(args.start_repeat, args.start_repeat + args.repeat_times):
    model_path = os.path.join(
        out_dir,
        "checkpoint",
        "{}_{}".format("_".join([str(v) for v in base_config_list]), repeat_idx),
    )

    if len(args.pretrained_cp_path) != 0:

        pretrained_cp_path_final = args.pretrained_cp_path.format(repeat_idx)
    else:
        pretrained_cp_path_final = ""
    model = Model(
        tra_config={
            "num_states": args.num_states,
            "hidden_size": args.tra_hidden_size,
            "tau": 1.0,
            "src_info": "LR_TPE",
        },
        model_config={
            "input_size": 6,
            "hidden_size": 64,
            "num_layers": 2,
            # "num_heads": args.rnn_num_heads,
            "use_attn": args.not_use_attn,
            "dropout": args.dropout,
        },
        model_type=args.model_type,
        lr=args.lr,
        n_epochs=500,
        max_steps_per_epoch=100,
        early_stop=args.early_stop,
        logdir=model_path,  #'output/Alpha360',
        seed=10000,
        lamb=2.0,
        rho=0.99,
        freeze_model=False,
        freeze_predictors=False,
        length_after_resample=args.length_after_resample,
        patterns=args.patterns,
        GPU=0,
    )
    model.fit(dataset)
    pred_result = model.predict(dataset)
    test_ic_results = [
        metric_fn_txnx1(pred_result[i], pred_result[-1])
        for i in range(len(pred_result) - 1)
    ]
    values_list = [args.model, repeat_idx]
    for test_ic_result in test_ic_results:
        values_list.extend(list(test_ic_result.values()))
    final_result.append(values_list)
    print(final_result)
    final_result_df = pd.DataFrame(final_result)
    columns = ["model", "repeat"]
    for test_ic_result in test_ic_results:
        columns.extend(list(test_ic_result.keys()))
    final_result_df.columns = columns
    final_result_df.to_csv(result_path)
