from .backend import get_backend

if get_backend() == 'tf':
    from .backend_tf import set_session

if get_backend() == 'torch':
    from .backend_torch import set_session

from .agent.dqn import DqnModel, DuelingDqnModel, DqnConvModel
from .agent.dqn.dqn_agent import DqnAgent
from .agent.random_agent import RandomAgent
from .agent.sarsa_agent import SarsaAgent
from .env import GymEnv, UnityEnv
from .simulator import Simulator
