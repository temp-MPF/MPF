# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import io
import os
import copy
import math
import json
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from tqdm import tqdm

from qlib.utils import get_or_create_path
from qlib.constant import EPS
from qlib.log import get_module_logger
from qlib.model.base import Model
from qlib.contrib.data.dataset import MTSDatasetH
import json
import sys
from scipy import signal

sys.path.append('../../')
from utils_src.torch_utils import gumbel_softmax
from utils_src.utils import metric_fn_txnx1

import collections
from tqdm import tqdm


# device = "cuda" if torch.cuda.is_available() else "cpu"


class MPF(Model):
    def __init__(
            self,
            model_config,
            tra_config,
            model_type="LSTM",
            lr=1e-3,
            n_epochs=500,
            early_stop=50,
            smooth_steps=5,
            max_steps_per_epoch=None,
            freeze_model=False,
            model_init_state=None,
            lamb=0.0,
            rho=0.99,
            seed=None,
            logdir=None,
            eval_train=True,
            eval_test=False,
            avg_params=True,
            patterns=json.dumps(['60', '-40:', '-30:', '::2']),
            length_after_resample=json.dumps([80, 60, 60, 60]),
            pretrained_checkpoint_list=json.dumps([]),
            GPU=0,
            fussion_model='fussion',
            is_freeze='true',
            optimizer="adam",
            base_lr=0.,
            head_lr=0.,
            base_weight_decay=0.001,
            head_weight_decay=0.001,
            branch_w = json.dumps([0.1,0.1,0.1,0.1]),
            **kwargs,
    ):

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.logger = get_module_logger("MPF")
        self.logger.info("MPF Model...")
        self.patterns = json.loads(patterns)
        self.length_after_resample = json.loads(length_after_resample)
        self.lstm_num = len(self.patterns)
        self.diff_len_num = self.lstm_num
        self.device = torch.device("cuda:%d" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.base_weight_decay = base_weight_decay
        self.head_weight_decay = head_weight_decay
        self.fussion_model = fussion_model
        self.is_freeze = is_freeze
        self.model = MPFModel(model_config,
                             tra_config,
                             model_type,
                             freeze_model=freeze_model,
                             model_init_state=model_init_state,
                             device=self.device,
                             logger=self.logger,
                             lstm_num=self.lstm_num,
                             fussion_model=self.fussion_model)
        self.branch_w = json.loads(branch_w)
        latent_params_id = []
        for t_model in self.model.lstm_tras:
            latent_params_id.extend(list(map(id, t_model.parameters())))

        base_params = filter(lambda x: id(x) in latent_params_id, self.model.parameters())
        latent_params = filter(lambda x: id(x) not in latent_params_id, self.model.parameters())
        if optimizer.lower() == "adam":
            self.optimizer = optim.Adam(
                [
                    {'params': base_params, 'lr': lr if base_lr <= 0. else base_lr,
                     'weight_decay': self.base_weight_decay},
                    {'params': latent_params, 'lr': lr if head_lr <= 0. else base_lr,
                     'weight_decay': self.head_weight_decay},
                ]
            )
        elif optimizer.lower() == "gd":
            self.optimizer = optim.SGD(
                [
                    {'params': base_params, 'lr': lr if base_lr <= 0. else base_lr,
                     'weight_decay': self.base_weight_decay},
                    {'params': latent_params, 'lr': lr if head_lr <= 0. else base_lr,
                     'weight_decay': self.head_weight_decay},
                ]
            )

        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))
        self.model_config = model_config
        self.tra_config = tra_config
        self.lr = lr
        self.n_epochs = n_epochs
        self.early_stop = early_stop
        self.smooth_steps = smooth_steps
        self.max_steps_per_epoch = max_steps_per_epoch
        self.lamb = lamb
        self.rho = rho
        self.seed = seed
        self.logdir = logdir
        self.eval_train = eval_train
        self.eval_test = eval_test
        self.avg_params = avg_params

        if self.model.lstm_tras[0].tra.num_states > 1 and not self.eval_train:
            self.logger.warn("`eval_train` will be ignored when using TRA")

        if self.logdir is not None:
            if os.path.exists(self.logdir):
                self.logger.warn(f"logdir {self.logdir} is not empty")
            os.makedirs(self.logdir, exist_ok=True)

        self.fitted = False
        self.global_step = -1
        self.pretrained_checkpoint_list = json.loads(pretrained_checkpoint_list)
        self.load_checkpoint(self.pretrained_checkpoint_list)
        if self.is_freeze.lower() == 'true':
            print("the base model will be freezed!")
            for param in self.model.lstm_tras.parameters():
                param.requires_grad_(False)

    def load_checkpoint(self, pretrained_checkpoint_list):
        for i in range(len(pretrained_checkpoint_list)):
            if pretrained_checkpoint_list[i] is not None and len(pretrained_checkpoint_list[i]) != 0 and os.path.exists(
                    pretrained_checkpoint_list[i]):
                self.model.lstm_tras[i].model.load_state_dict(
                    torch.load(pretrained_checkpoint_list[i], map_location="cpu")['model'])
                self.model.lstm_tras[i].tra.load_state_dict(
                    torch.load(pretrained_checkpoint_list[i], map_location="cpu")['tra'])
                print("successfully loading pretrained checkpoint from {}".format(pretrained_checkpoint_list[i]))

    def train_epoch(self, data_set):

        self.model.train()

        data_set.train()

        max_steps = self.n_epochs
        if self.max_steps_per_epoch is not None:
            max_steps = min(self.max_steps_per_epoch, self.n_epochs)

        count = 0
        total_loss = 0
        total_count = 0
        for batch in tqdm(data_set, total=max_steps):
            count += 1
            if count > max_steps:
                break

            self.global_step += 1

            data, state, label, count = batch["data"], batch["state"], batch["label"], batch["daily_count"]
            index = batch["index"]

            x_train_values_list = []
            for i in range(len(self.patterns)):
                x_train_values_new = eval("data.detach().cpu().numpy()[:,{},:]".format(self.patterns[i]))
                x_train_values_list.append(signal.resample(x_train_values_new, self.length_after_resample[i], axis=1))
            batch_x_train_values = [torch.tensor(x_train_values_list[j]).float().to(self.device) for j in
                                    range(self.diff_len_num)]

            outs, out_dynamic, out_static, out_final = self.model(batch_x_train_values, state.to(self.device))
            # pred, all_preds, prob = self.tra(hidden, state)
            outs_pred = torch.stack( [v[0] for v in outs], dim=-1)  # batch x lstm_num
            branch_w = torch.tensor( self.branch_w, dtype = outs_pred.dtype).to(self.device)
            loss = ((out_final - label.to(self.device)).pow(2).mean() + 
                    ((outs_pred - label[:, None].to(self.device)).pow(2).mean(dim=0)*branch_w).sum() ) * 1.0 / (
                               1. + branch_w.sum() )

            all_preds = [outs[branchi][1] for branchi in range(self.lstm_num)]

            Ls = [(all_preds[i] - label[:, None].to(self.device)).pow(2) for i in range(self.lstm_num)]
            Ls = [Ls[i] - Ls[i].min(dim=-1, keepdim=True).values for i in
                  range(self.lstm_num)]  # normalize & ensure positive input

            data_set.assign_data(index, Ls)  # save loss to memory
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()
            total_count += len(out_final)

        total_loss /= total_count

        return total_loss

    def test_epoch(self, data_set, return_pred=False):

        self.model.eval()
        data_set.eval()

        preds = []
        metrics = []
        for batch in tqdm(data_set):
            data, state, label, count = batch["data"], batch["state"], batch["label"], batch["daily_count"]
            index = batch["index"]
            x_train_values_list = []
            for i in range(len(self.patterns)):
                x_train_values_new = eval("data.detach().cpu().numpy()[:,{},:]".format(self.patterns[i]))
                x_train_values_list.append(
                    signal.resample(x_train_values_new, self.length_after_resample[i], axis=1))
            batch_x_train_values = [torch.tensor(x_train_values_list[j]).float().to(self.device) for j in
                                    range(self.diff_len_num)]
            with torch.no_grad():
                outs, out_dynamic, out_static, out_final = self.model(batch_x_train_values, state.to(self.device))
            
            all_preds = [outs[branchi][1] for branchi in range(self.lstm_num)]

            Ls = [(all_preds[i] - label[:, None].to(self.device)).pow(2) for i in range(self.lstm_num)]
            Ls = [Ls[i] - Ls[i].min(dim=-1, keepdim=True).values for i in
                  range(self.lstm_num)]  # normalize & ensure positive input

            data_set.assign_data(index, Ls)  # save loss to memory

            X = np.c_[
                out_final.cpu().numpy(),
                label.cpu().numpy(),
            ]
            columns = ["score", "label"]
            pred = pd.DataFrame(X, index=index, columns=columns)
            metrics.append(evaluate(pred))
            if return_pred:
                preds.append(pred)

        metrics = pd.DataFrame(metrics)
        metrics = {
            "MSE": metrics.MSE.mean(),
            "MAE": metrics.MAE.mean(),
            "IC": metrics.IC.mean(),
            "ICIR": metrics.IC.mean() / metrics.IC.std(),
        }

        if return_pred:
            preds = pd.concat(preds, axis=0)
            preds.index = data_set.restore_index(preds.index)
            preds.index = preds.index.swaplevel()
            preds.sort_index(inplace=True)

        return metrics, preds

    def fit(self, dataset, evals_result=dict()):

        train_set, valid_set, test_set = dataset.prepare(["train", "valid", "test"])

        best_score = -1
        best_epoch = 0
        stop_rounds = 0
        best_params = {
            "model": copy.deepcopy(self.model.state_dict()),
        }
        params_list = {
            "model": collections.deque(maxlen=self.smooth_steps),
        }
        evals_result["train"] = []
        evals_result["valid"] = []
        evals_result["test"] = []

        # train
        self.fitted = True
        self.global_step = -1

        if self.model.lstm_tras[0].tra.num_states > 1:
            self.logger.info("init memory...")
            self.test_epoch(train_set)

        for epoch in range(self.n_epochs):
            self.logger.info("Epoch %d:", epoch)

            self.logger.info("training...")
            self.train_epoch(train_set)

            self.logger.info("evaluating...")
            # average params for inference
            params_list["model"].append(copy.deepcopy(self.model.state_dict()))
            self.model.load_state_dict(average_params(params_list["model"]))

            # NOTE: during evaluating, the whole memory will be refreshed
            if self.model.lstm_tras[0].tra.num_states > 1 or self.eval_train:
                train_set.clear_memory()  # NOTE: clear the shared memory
                train_metrics = self.test_epoch(train_set)[0]
                evals_result["train"].append(train_metrics)
                self.logger.info("\ttrain metrics: %s" % train_metrics)

            valid_metrics = self.test_epoch(valid_set)[0]
            evals_result["valid"].append(valid_metrics)
            self.logger.info("\tvalid metrics: %s" % valid_metrics)

            if self.eval_test:
                test_metrics = self.test_epoch(test_set)[0]
                evals_result["test"].append(test_metrics)
                self.logger.info("\ttest metrics: %s" % test_metrics)

            if valid_metrics["IC"] > best_score:
                best_score = valid_metrics["IC"]
                stop_rounds = 0
                best_epoch = epoch
                best_params = {
                    "model": copy.deepcopy(self.model.state_dict()),
                }
            else:
                stop_rounds += 1
                if stop_rounds >= self.early_stop:
                    self.logger.info("early stop @ %s" % epoch)
                    break

            # restore parameters
            self.model.load_state_dict(params_list["model"][-1])

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.model.load_state_dict(best_params["model"])

        metrics, preds = self.test_epoch(test_set, return_pred=True)
        self.logger.info("test metrics: %s" % metrics)

        if self.logdir:
            self.logger.info("save model & pred to local directory")

            pd.concat({name: pd.DataFrame(evals_result[name]) for name in evals_result}, axis=1).to_csv(
                self.logdir + "/logs.csv", index=False
            )

            torch.save(best_params, self.logdir + "/model.bin")

            preds.to_pickle(self.logdir + "/pred.pkl")

            info = {
                "config": {
                    "model_config": self.model_config,
                    "tra_config": self.tra_config,
                    "lr": self.lr,
                    "n_epochs": self.n_epochs,
                    "early_stop": self.early_stop,
                    "smooth_steps": self.smooth_steps,
                    "max_steps_per_epoch": self.max_steps_per_epoch,
                    "lamb": self.lamb,
                    "rho": self.rho,
                    "seed": self.seed,
                    "logdir": self.logdir,
                },
                "best_eval_metric": -best_score,  # NOTE: minux -1 for minimize
                "metric": metrics,
            }
            with open(self.logdir + "/info.json", "w") as f:
                json.dump(info, f)

    def predict(self, dataset, segment="test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        test_set = dataset.prepare(segment)

        metrics, preds = self.test_epoch(test_set, return_pred=True)
        self.logger.info("test metrics: %s" % metrics)
        return [preds['score'], preds['label']]


class LSTM(nn.Module):
    """LSTM Model

    Args:
        input_size (int): input size (# features)
        hidden_size (int): hidden size
        num_layers (int): number of hidden layers
        use_attn (bool): whether use attention layer.
            we use concat attention as https://github.com/fulifeng/Adv-ALSTM/
        dropout (float): dropout rate
        input_drop (float): input dropout for data augmentation
        noise_level (float): add gaussian noise to input for data augmentation
    """

    def __init__(
            self,
            input_size=16,
            hidden_size=64,
            num_layers=2,
            use_attn=True,
            dropout=0.0,
            input_drop=0.0,
            noise_level=0.0,
            *args,
            **kwargs,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attn = use_attn
        self.noise_level = noise_level

        self.input_drop = nn.Dropout(input_drop)

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        if self.use_attn:
            self.W = nn.Linear(hidden_size, hidden_size)
            self.u = nn.Linear(hidden_size, 1, bias=False)
            self.output_size = hidden_size * 2
        else:
            self.output_size = hidden_size

    def forward(self, x):

        x = self.input_drop(x)

        if self.training and self.noise_level > 0:
            noise = torch.randn_like(x).to(x)
            x = x + noise * self.noise_level
        # print (x.shape)
        rnn_out, _ = self.rnn(x)
        last_out = rnn_out[:, -1]

        if self.use_attn:
            laten = self.W(rnn_out).tanh()
            scores = self.u(laten).softmax(dim=1)
            att_out = (rnn_out * scores).sum(dim=1).squeeze()
            last_out = torch.cat([last_out, att_out], dim=1)

        return last_out




class TRA(nn.Module):
    """Temporal Routing Adaptor (TRA)

    TRA takes historical prediction errors & latent representation as inputs,
    then routes the input sample to a specific predictor for training & inference.

    Args:
        input_size (int): input size (RNN/Transformer's hidden size)
        num_states (int): number of latent states (i.e., trading patterns)
            If `num_states=1`, then TRA falls back to traditional methods
        hidden_size (int): hidden size of the router
        tau (float): gumbel softmax temperature
    """

    def __init__(self, input_size, num_states=1, hidden_size=8, tau=1.0, src_info="LR_TPE"):
        super().__init__()

        self.num_states = num_states
        self.tau = tau
        self.src_info = src_info

        if num_states > 1:
            self.router = nn.LSTM(
                input_size=num_states,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
            )
            self.fc = nn.Linear(hidden_size + input_size, num_states)

        self.predictors = nn.Linear(input_size, num_states)

    def forward(self, hidden, hist_loss):

        preds = self.predictors(hidden)

        if self.num_states == 1:
            return preds.squeeze(-1), preds, None

        # information type
        router_out, _ = self.router(hist_loss)
        if "LR" in self.src_info:
            latent_representation = hidden
        else:
            latent_representation = torch.randn(hidden.shape).to(hidden)
        if "TPE" in self.src_info:
            temporal_pred_error = router_out[:, -1]
        else:
            temporal_pred_error = torch.randn(router_out[:, -1].shape).to(hidden)
        # print (router_out.shape,temporal_pred_error.shape,latent_representation.shape)
        out = self.fc(torch.cat([temporal_pred_error, latent_representation], dim=-1))
        prob = F.gumbel_softmax(out, dim=-1, tau=self.tau, hard=False)

        if self.training:
            final_pred = (preds * prob).sum(dim=-1)
        else:
            final_pred = preds[range(len(preds)), prob.argmax(dim=-1)]

        return final_pred, preds, prob


class LSTM_TRA(nn.Module):
    def __init__(self,
                 model_config,
                 tra_config,
                 model_type,
                 freeze_model=False,
                 model_init_state=None,
                 device=None,
                 logger=None,

                 ):
        super().__init__()
        self.logger = logger
        self.device = device
        self.model = eval(model_type)(**model_config).to(self.device)
        if model_init_state:
            self.model.load_state_dict(torch.load(model_init_state, map_location="cpu")["model"])
        if freeze_model:
            for param in self.model.parameters():
                param.requires_grad_(False)
        else:
            self.logger.info("# model params: %d" % sum([p.numel() for p in self.model.parameters()]))

        self.tra = TRA(self.model.output_size, **tra_config).to(self.device)

    def forward(self, x, state):
        hidden = self.model(x)
        pred, all_preds, prob = self.tra(hidden, state)
        return [pred, all_preds, prob, hidden]  #


class MPFModel(nn.Module):
    def __init__(self,
                 model_config,
                 tra_config,
                 model_type,
                 freeze_model=False,
                 model_init_state=None,
                 device=None,
                 logger=None,
                 lstm_num=1,
                 fussion_model="fussion",
                 ):
        super().__init__()
        self.fussion_model = fussion_model
        self.lstm_num = lstm_num
        self.lstm_tras = nn.ModuleList([LSTM_TRA(
            model_config,
            tra_config,
            model_type,
            freeze_model,
            model_init_state,
            device,
            logger
        ) for _ in range(self.lstm_num)])
        self.static_w = torch.nn.Parameter(torch.rand(self.lstm_num, 1), requires_grad=True)
        self.static_w.data.normal_(0, 1)
        self.dynamic_fussion_a = nn.Linear(self.lstm_num, 1).to(device)
        print("the fussion_model is {}".format(self.fussion_model))

    def forward(self, xs, states):
        outs = [self.lstm_tras[i](xs[i], states[:, i]) for i in range(self.lstm_num)]
        outs_pred = torch.stack([v[0] for v in outs], dim=-1)  # batch x lstm_num
        hidden = torch.stack([v[3] for v in outs], dim=1)  # batch  x lstm_num  x feature_dim
        out_static = torch.matmul(outs_pred, torch.softmax(self.static_w, dim=0).to(outs_pred.device))  # lstm_num x 1
        dynamic_w1 = torch.einsum('nbf,nfa->nba', hidden, hidden.permute(0, 2, 1))
        dynamic_w1 = (dynamic_w1 - dynamic_w1.min(dim=-1, keepdim=True)[0]) / (
                    dynamic_w1.max(dim=-1, keepdim=True)[0] - dynamic_w1.min(dim=-1, keepdim=True)[0] + 1e-8)
        dynamic_w = torch.softmax(self.dynamic_fussion_a(dynamic_w1.permute(0, 2, 1)).squeeze(-1),
                                  dim=-1)  # batch x lstm_num
        #         dynamic_w = torch.softmax(self.dynamic_fussion(hidden).squeeze(-1),dim = -1) # batch x lstm_num
        out_dynamic = torch.sum(outs_pred * dynamic_w, dim=-1, keepdim=True)
        if self.fussion_model == "fussion":
            return outs, out_dynamic, out_static, (out_dynamic + out_static) / 2
        elif self.fussion_model == "static":
            return outs, out_dynamic, out_static, out_static
        return outs, out_dynamic, out_static, out_dynamic  # out_static#out_dynamic#(out_dynamic+out_static)/2


def evaluate(pred):
    pred = pred.rank(pct=True)  # transform into percentiles
    score = pred.score
    label = pred.label
    diff = score - label
    MSE = (diff ** 2).mean()
    MAE = (diff.abs()).mean()
    IC = score.corr(label)
    return {"MSE": MSE, "MAE": MAE, "IC": IC}


def average_params(params_list):
    assert isinstance(params_list, (tuple, list, collections.deque))
    n = len(params_list)
    if n == 1:
        return params_list[0]
    new_params = collections.OrderedDict()
    keys = None
    for i, params in enumerate(params_list):
        if keys is None:
            keys = params.keys()
        for k, v in params.items():
            if k not in keys:
                raise ValueError("the %d-th model has different params" % i)
            if k not in new_params:
                new_params[k] = v / n
            else:
                new_params[k] += v / n
    return new_params

