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
brew install python3 swig && \
    brew install opencv3 --with-python && \
    pip3 install --upgrade pip setuptools wheel
```

##### Ubuntu
```shell
sudo apt-get install swig python3 python3-pip && \
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
***

<br/><br/><br/>

# Environments

## Banana (Unity)

![Watch](reports/banana/2018-09-02.gif)

#### Environment Details

* __Type:__ UnityML
* __Goal:__ The agents must learn to collect as many yellow bananas as possible while avoiding blue bananas.
* __Reward:__
   - +1 for collecting yellow banana.
   - -1 for collecting blue banana
* __State Space:__  37 dimensions which contains the agent's velocity, along with ray-based perception of objects around 
agent's forward direction.
* __Action Space:__
   - __0__ - move forward.
   - __1__ - move backward.
   - __2__ - turn left.
   - __3__ - turn right.
* __Solved when:__ Agent gets an average score of +13 over 100 consecutive episodes.

#### Setup
1. Download the environment from one of the links below to the checked out directory. You need only select the environment that matches your operating system:
   - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
   - MacOS: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
   - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
   - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
2. Edit [configs/banana.yaml](configs/banana.yaml) and change `filename` field according to your environment.
   - Linux: `Banana_Linux/Banana.x86_64`
   - MacOS: `Banana.app`
   - Windows (32-bit): `Banana_Windows_x86/Banana.exe`
   - Windows (64-bit): `Banana_Windows_x86_64/Banana.exe`

#### Usage

##### Train
```shell
python3 train.py -c configs/banana.yaml
```

##### Watch
```shell
python3 watch.py -c configs/banana.yaml
```

## Credits 

### Reinforcement Learning base Implementation

* [OpenAi Gym Environment](https://github.com/openai/gym)
* [Unity Machine Learning Agents Toolkit](https://github.com/Unity-Technologies/ml-agents)
* [Udacity Deep Reinforcement Nano Degree Agent Simulator](https://github.com/udacity/deep-reinforcement-learning/blob/master/lab-taxi/monitor.py)
* [Jaromír Janisch's SumTree Implementation](https://github.com/jaara/AI-blog/blob/master/SumTree.py)

### Machine Learning Softwares

* [TensorFlow](https://github.com/tensorflow/tensorflow)
* [PyTorch](https://github.com/pytorch/pytorch)
* [Keras](https://github.com/keras-team/keras)
