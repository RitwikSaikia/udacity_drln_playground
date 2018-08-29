Inspired by the [Deep Reinforcement Learning Nanodegree](https://udacity.com/course/deep-learning-nanodegree--nd101) course material.


## Summary

* [Switch backends](#switch-backends) between **Tensorflow** and **PyTorch**.
* Explore new environment via yaml [config files](configs/) (no coding required).
* Supported Algorithms
   - Expected Sarsa
   - [Deep Q Learning](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf)
   - [Dueling Deep Q Learning](https://arxiv.org/abs/1511.06581)
   - [Double Deep Q Learning](https://arxiv.org/abs/1509.06461)
   - [Prioritized Experience Replay (with Importance Sampling)](https://arxiv.org/abs/1511.05952)
   - CNN variation of **Deep Q Network** and **Dueling Deep Q Network** 
* Supported Environments
   - [OpenAI Gym](https://github.com/openai/gym)
   - [Unity ML Agents](https://github.com/Unity-Technologies/ml-agents)
* [Pre-trained Checkpoints](checkpoints/) for few environments.

## Installation

#### 1. Setup Python 3 

##### MacOS
```shell
brew install python3 && \
    brew install opencv3 --with-python && \
    pip3 install --upgrade pip setuptools wheel
```

##### Ubuntu
```shell
sudo apt-get install python3 python3-pip && \
    sudo pip3 install --upgrade pip setuptools wheel && \
    sudo pip3 install virtualenv
```

#### 2. Setup Virtual Environment
```shell
virtualenv --no-site-packages -p python3 .venv && \
    source .venv/bin/activate && \
    pip install -r requirements.txt
```

## Usage

#### 1. Switch to Virtual Environment 
```shell
source .venv/bin/activate
```

#### 2. Train an Agent
```shell
python3 train.py -c configs/cartpole.yaml
```

#### 3. Watch an Agent
```shell
python3 watch.py -c configs/cartpole.yaml
```


## [Switch Backends](#switch-backends)

By default **PyTorch** backend is used, if it is available in the runtime. 
In case you want to switch to:
##### Tensorflow
```shell
export RL_BACKEND='tf'
``` 

##### PyTorch
```shell
export RL_BACKEND='torch'
``` 
