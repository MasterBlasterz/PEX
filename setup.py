from setuptools import setup, find_packages

setup(
    name='pex',
    version='0.0.1',
    python_requires='>=3.11,<3.12',
    install_requires=[
        # 'tqdm',
        # 'scipy',
        # 'pandas',
        # 'torch>=1.8.1',
        # 'gym==0.15.4',
        # 'd4rl@git+https://github.com/rail-berkeley/d4rl@master#egg=d4rl',
        'tqdm==4.68.3',
        'torch==2.5.1',
        'gymnasium==1.3.0',
        'minari==0.5.2',
        'mujoco==3.2.3',
        'minari[all]',
        'imageio',
        'wandb',
        'weave',
    ],
    packages=find_packages(),
)
