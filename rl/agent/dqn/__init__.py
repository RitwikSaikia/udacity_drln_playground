from ...backend import get_backend

if get_backend() == 'tf':
    from .dqn_model_tf import DqnModel, DuelingDqnModel, DqnConvModel, DuelingDqnConvModel

if get_backend() == 'torch':
    from .dqn_model_torch import DqnModel, DuelingDqnModel, DqnConvModel, DuelingDqnConvModel
