# Policy Expansion (PEX)  [ICLR 2023]

## Installation
The training environment (PyTorch and dependencies) can be installed as follows:

```
git clone https://github.com/Haichao-Zhang/PEX.git
cd PEX

python3 -m venv .venv_pex
source .venv_pex/bin/activate
pip3 install -e .
```

## Installation (nr3)
```
git clone https://github.com/MasterBlasterz/PEX.git
cd PEX
conda create -n <ENV> python=3.11
conda activate <ENV>
pip install -e .
```

### Environment Changes
New Minari Names:
- mujoco/halfcheetah/random-v0
- mujoco/halfcheetah/medium-v0
- mujoco/halfcheetah/medium-replay-v0
- mujoco/hopper/random-v0
- mujoco/hopper/medium-v0
- mujoco/hopper/medium-replay-v0
- mujoco/walker2d/random-v0
- mujoco/walker2d/medium-v0
- mujoco/walker2d/medium-replay-v0


## Train

### Offline Training

Set ```root_dir``` to the path where the experimental results will be saved.


#### Set ENV Vars

export PROJECT_PATH="/project/dl2026s/${USER}"
export root_dir=$PROJECT_PATH/logs

mkdir -p $root_dir
mkdir -p $PROJECT_PATH/{hf_cache,minari}

export HF_HOME=$PROJECT_PATH/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export MINARI_DATASETS_PATH=$PROJECT_PATH/minari

Then choose the environment:

```
export env_name=mujoco/walker2d/medium-v0
export log_dir=$root_dir/walker2d_medium_v0_run1
export num_steps=100
```

Then run:

```
CUDA_VISIBLE_DEVICES=0 python main_offline.py --log_dir=$log_dir --env_name=$env_name --tau 0.9 --beta 10.0 --num_steps $num_steps
```
**NOTE**: For only a quick run add the arguments `--num-steps=<NUMSTEPS>` and `--eval_period=<EVALPERIOD>`.

### Online Training
First set the path to the offline checkpoint:

and select an algorithm:
```
export algorithm=pex (or any other algorithms in [scratch, direct, buffer, pex])
```

and then run
```
CUDA_VISIABLE_DEVICES=0 python ./main_online.py --log_dir=${log_dir}_online --env_name=$env_name --tau 0.9 --beta 10.0 --eval_episode_num=10 --algorithm=$algorithm --ckpt_path=$log_dir/offline_ckpt
```

echo $LD_PRELOAD



## Paper

<b>[Policy Expansion for Bridging Offline-to-Online Reinforcement Learning](https://arxiv.org/pdf/2302.00935.pdf)</b> <br>

[Haichao Zhang](https://sites.google.com/site/hczhang1/),
Wei Xu,
Haonan Yu

*International Conference on Learning Representations* (ICLR), 2023



## Cite

Please cite our work if you find it useful:

```
@inproceedings{PEX,
  author    = {Haichao Zhang and Wei Xu and Haonan Yu},
  title     = {Policy Expansion for Bridging Offline-to-Online Reinforcement Learning},
  booktitle = {International Conference on Learning Representations ({ICLR})},
  year      = {2023},
}
```
