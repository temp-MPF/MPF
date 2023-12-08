# MPF

## prepare the environment 

1. create conda environment
    ```bash
    conda create --name mpf python=3.8 
    conda activate mpf
    ```

2. install python package 
    ```bash
    pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu116
    ```

3. install pyqlib
    - download source code from  [web](https://github.com/microsoft/qlib/releases/tag/mini_projectV01) or [url](https://github.com/microsoft/qlib/archive/refs/tags/mini_projectV01.tar.gz).
    - install pyqlib 
        ```bash
        tar -zxvf qlib-mini_projectV01.tar.gz  && cd qlib-mini_projectV01
        pip install --upgrade  cython 
        pip install .
        ```

## download the data 
- Load and prepare data by running the following code.
    ```bash 
    cd qlib-mini_projectV01 
    python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn
    ```
- the more information about the data can be found in [qlib](https://github.com/microsoft/qlib/tree/mini_projectV01) .

    

## train the model 

1. make directory
 
    ```bash
    mkdir results_single
    mkdir results_mpf
    ```

3. train the single pattern model 
    - train the single pattern $p_5=(0.5\lambda ,2l)$ model on CSI100
    ```bash 
    nohup python -u train_single.py     --model 'single120'     --instruments 'csi100'     --out_dir './results_single'     --gpu_id 0     --repeat_times 10     --time_step 60     --n_features 6     --data_dir '~/.qlib/qlib_data/cn_data'     --pattern '[":"]'     --num_states 1     --dropout 0.1       --rnn_hidden_size 64     --tra_hidden_size 32     --length_after_resample '[120]'   --lr 0.001  --model_type "LSTM" > "results_single/single120_lr1e3_LSTM_num_states1"  2>&1 &
    ```
    
    - train the single pattern $p_4=(\lambda ,l)$ model  on CSI100
    ```bash
    nohup python -u train_single.py     --model 'single60'     --instruments 'csi100'     --out_dir './results_single'     --gpu_id 0     --repeat_times 1     --time_step 60     --n_features 6     --data_dir '~/.qlib/qlib_data/cn_data'     --pattern '[":"]'     --num_states 1     --dropout 0.1         --rnn_hidden_size 64     --tra_hidden_size 32     --length_after_resample '[60]'   --lr 0.0001  --model_type "LSTM" > "results_single/single60_le4_LSTM_num_states1"  2>&1 &
    ```
    
    - train the single pattern $p_3=(2\lambda ,0.5l)$ model  on CSI100
    ```bash
    nohup python -u train_single.py     --model 'single30'     --instruments 'csi100'     --out_dir './results_single'     --gpu_id 1     --repeat_times 1     --time_step 60     --n_features 6     --data_dir '~/.qlib/qlib_data/cn_data'     --pattern '[":"]'     --num_states 1     --dropout 0.1         --rnn_hidden_size 64     --tra_hidden_size 32     --length_after_resample '[30]'   --lr 0.0001  --model_type "LSTM" > "results_single/single30_le4_LSTM_num_states1"  2>&1 &
    ```
    
    - train the single pattern $p_2=(\frac{2}{3}\lambda , l)$ model  on CSI100
    ```bash
    nohup python -u train_single.py     --model 'single4060'     --instruments 'csi100'     --out_dir './results_single'     --gpu_id 2     --repeat_times 1     --time_step 60     --n_features 6     --data_dir '~/.qlib/qlib_data/cn_data'     --pattern '["-40:"]'     --num_states 1     --dropout 0.1         --rnn_hidden_size 64     --tra_hidden_size 32     --length_after_resample '[60]'  --lr 0.001 --model_type "LSTM" > "results_single/single4060_le3_LSTM_num_states1"  2>&1 &
    ```

    - train the single pattern $p_1=(0.5\lambda , l)$ model  on CSI100
    ```bash
    nohup python -u train_single.py     --model 'single3060'     --instruments 'csi100'     --out_dir './results_single'     --gpu_id 3     --repeat_times 1     --time_step 60     --n_features 6     --data_dir '~/.qlib/qlib_data/cn_data'     --pattern '["-30:"]'     --num_states 1     --dropout 0.1          --rnn_hidden_size 64     --tra_hidden_size 32     --length_after_resample '[60]'  --lr 0.0001  --model_type "LSTM"> "results_single/single3060_le4_LSTM_num_states1"  2>&1 &
    ```
    
    - train the single pattern $p_5=(0.5\lambda ,2l)$ model   on CSI500
    ```bash
    nohup python -u train_single.py     --model 'single120'     --instruments 'csi500'     --out_dir './results_single'     --gpu_id 0     --repeat_times 1     --time_step 60     --n_features 6     --data_dir '~/.qlib/qlib_data/cn_data'   --pattern '[":"]'     --num_states 3     --dropout 0.1       --rnn_hidden_size 64     --tra_hidden_size 32     --length_after_resample '[120]'   --lr 0.0001  --model_type "LSTM"  > "results_single/single120_lr1e4_LSTM_num_states3_csi500"  2>&1 &
    ```

    - train the single pattern $p_4=(\lambda ,l)$ model  on CSI500
    ```bash
    nohup python -u train_single.py     --model 'single60'     --instruments 'csi500'     --out_dir './results_single'     --gpu_id 0     --repeat_times 1     --time_step 60     --n_features 6     --data_dir '~/.qlib/qlib_data/cn_data'     --pattern '[":"]'     --num_states 3     --dropout 0.1     --rnn_hidden_size 64     --tra_hidden_size 32     --length_after_resample '[60]'   --lr 0.0001  --model_type "LSTM"   > "results_single/single60_le4_LSTM_num_states3_csi500"  2>&1 &
    ```

    - train the single pattern $p_3=(2\lambda ,0.5l)$ model  on CSI500
    ```bash
    nohup python -u train_single.py     --model 'single30'     --instruments 'csi500'     --out_dir './results_single'     --gpu_id 2     --repeat_times 1     --time_step 60     --n_features 6     --data_dir '~/.qlib/qlib_data/cn_data'     --pattern '[":"]'     --num_states 3     --dropout 0.1       --rnn_hidden_size 64     --tra_hidden_size 32     --length_after_resample '[30]'   --lr 0.0001  --model_type "LSTM"  > "results_single/single30_le4_LSTM_num_states3_csi500"  2>&1 &
    ```

    -  train the single pattern $p_2=(\frac{2}{3}\lambda , l)$ model  on CSI500
    ```bash
    nohup python -u train_single.py     --model 'single3060'     --instruments 'csi500'     --out_dir './results_single'     --gpu_id 3     --repeat_times 1     --time_step 60     --n_features 6     --data_dir '~/.qlib/qlib_data/cn_data'     --pattern '["-30:"]'     --num_states 3     --dropout 0.1     --rnn_hidden_size 64     --tra_hidden_size 32     --length_after_resample '[60]'  --lr 0.0001  --model_type "LSTM"  > "results_single/single3060_le4_LSTM_num_states3_csi500"  2>&1 &
    ```

    - train the single pattern $p_1=(0.5\lambda , l)$ model  on CSI500
    ```bash
    nohup python -u train_single.py     --model 'single4060'     --instruments 'csi500'     --out_dir './results_single'     --gpu_id 1     --repeat_times 1     --time_step 60     --n_features 6     --data_dir '~/.qlib/qlib_data/cn_data'     --pattern '["-40:"]'     --num_states 3     --dropout 0.1    --rnn_hidden_size 64     --tra_hidden_size 32     --length_after_resample '[60]'  --lr 0.0001 --model_type "LSTM"  > "results_single/single4060_le4_LSTM_num_states3_csi500"  2>&1 &
        ```

4. train the mpf
    
    - train the mpf on CSI100
    ```bash
    nohup python -u train_mpf.py --model mpf --instruments csi100 --out_dir ./results_mpf --gpu_id 0 --repeat_times 1 --time_step 60 --n_features 6 --data_dir ~/.qlib/qlib_data/cn_data --lr 0.0001 --num_states 1 --tra_hidden_size 32  --dropout 0.1 --early_stop 20 --length_after_resample [120,60,30,60,60] --patterns '[":",":",":","-40:","-30:"]' --pretrained_cp_path "[\"./results_single/checkpoint/single120_csi100_60_6_0.001_256_0_10_1_LSTM_True_{}/model.bin\",\"./results_single/checkpoint/single60_csi100_60_6_0.0001_256_0_1_1_LSTM_True_{}/model.bin\",\"./results_single/checkpoint/single30_csi100_60_6_0.0001_256_0_1_1_LSTM_True_{}/model.bin\",\"./results_single/checkpoint/single4060_csi100_60_6_0.001_256_0_1_1_LSTM_True_{}/model.bin\",\"./results_single/checkpoint/single3060_csi100_60_6_0.0001_256_0_1_1_LSTM_True_{}/model.bin\"]"  --fussion_model "fussion"  --is_freeze "false" --head_weight_decay 0.001 --base_weight_decay 0. --base_lr 0.0001 --head_lr 0.0001 --branch_w "[1.0,1.0,1.0,0.2,0.2]" --model_type "LSTM" > "results_mpf/mpf_csi100_1e4-120-60-30-4060-3060_fussion_hwd1e3_bwd1e0_false_hlr1e4_blr1e4_10101022"  2>&1  &
    ```
    
    - train the mpf on CSI500
    ```bash 
    nohup python -u  train_mpf.py  --model "mpf"  --instruments "csi500"     --out_dir "./results_mpf"     --gpu_id 0     --repeat_times 1     --time_step 60     --n_features 6     --data_dir "~/.qlib/qlib_data/cn_data"     --lr 0.0001     --num_states 3   --tra_hidden_size 32          --dropout  0.1     --early_stop 20     --length_after_resample "[120,60,30,60,60]"  --patterns  "[\":\",\":\",\":\",\"-40:\",\"-30:\"]"  --pretrained_cp_path "[\"./results_single/checkpoint/single120_csi500_60_6_0.0001_256_0_1_3_LSTM_True_{}/model.bin\",\"./results_single/checkpoint/single60_csi500_60_6_0.0001_256_0_1_3_LSTM_True_{}/model.bin\",\"./results_single/checkpoint/single30_csi500_60_6_0.0001_256_0_1_3_LSTM_True_{}/model.bin\",\"./results_single/checkpoint/single4060_csi500_60_6_0.0001_256_0_1_3_LSTM_True_{}/model.bin\",\"./results_single/checkpoint/single3060_csi500_60_6_0.0001_256_0_1_3_LSTM_True_{}/model.bin\"]"  --fussion_model "fussion"  --is_freeze "false" --head_weight_decay 0.001 --base_weight_decay 0. --base_lr 0.0001 --head_lr 0.0001 --branch_w "[1.0,1.0,1.0,0.2,0.2]" --model_type "LSTM" > "results_mpf/mpf_csi500_1e4-120-60-30_fussion_hwd1e3_bwd1e0_false_hlr1e4_blr1e4_10101022"  2>&1  &
    ```
