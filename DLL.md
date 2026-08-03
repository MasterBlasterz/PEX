# Policy Expansion (PEX)  [ICLR 2023]

## Installation
The training environment (PyTorch and dependencies) can be installed as follows:

```
git clone https://github.com/MasterBlasterz/PEX.git
cd PEX

python3.11 -m venv .venv_pex
source .venv_pex/bin/activate
pip3 install -e .
```

## Train

### Offline Training

Set ```root_dir``` to the path where the experimental results will be saved.

Then run:

```
python main_offline.py --env_name mujoco/halfcheetah/medium-v0 --eval_period 10000 --log_dir=$root_dir/halfcheetah-medium-v0_offline_run1
```

For reproduction of our results add ```--seed XXX``` to the call.

| Environment | Seed 1 | Seed 2 | Seed 3
| - | - | - | - |
mujoco/halfcheetah/medium-v0 | 1411558164 | 1616222849 | 805293550
mujoco/hopper/medium-v0 | 853310902 | 557957050 | 23153608
mujoco/walker2d/medium-v0 | 1109422146 | 259802625 | 1616222849
mujoco/humanoid/medium-v0 | 1798607118


### Online Training
First set the path to the offline checkpoint:
```
path_to_offline_ckpt=$root_dir/halfcheetah-medium-v0_offline_run1/offline_ckpt
```

and select an algorithm:
```
algorithm=pex (or any other algorithms in [scratch, direct, buffer, pex])
```

and then run
```
python ./main_online.py --log_dir=$root_dir/halfcheetah-medium-v0_online_run1_$algorithm --env_name=mujoco/halfcheetah/medium-v0 --ckpt_path=$path_to_offline_ckpt --algorithm=$algorithm
```

For reproduction of our results add ```--seed XXX``` to the call.

| Environment | Seed 1 | Seed 2 | Seed 3
| - | - | - | - |
mujoco/halfcheetah/medium-v0 | 3743118481 | 2274139555 | 2824913460
mujoco/hopper/medium-v0 | 3743118481 | 2274139555 | 2824913460
mujoco/walker2d/medium-v0 | 3743118481 | 2274139555 | 2824913460
mujoco/humanoid/medium-v0 | 4257710153 | 2003530388

## Ablations

### Uniform Choice 

Add ```--ablation uni``` to activate the ablation.

### Humanoid Ablation

Set the algorithm variable to ```pex_grouped```.