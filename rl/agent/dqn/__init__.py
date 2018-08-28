from ...backend import get_backend

if get_backend() == 'tf':
    from .dqn_model_tf import DqnModel, DuelingDqnModel, DqnConvModel
