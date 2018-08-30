import os
import random
import sys

import numpy as np

_BACKEND = None
ALLOWED_BACKENDS = {
    'tf': 'Tensorflow',
    'torch': 'Torch'
}


def get_backend():
    global _BACKEND

    if 'RL_BACKEND' in os.environ.keys() and _BACKEND is None:
        env_backend = os.environ['RL_BACKEND']
        if env_backend not in ALLOWED_BACKENDS:
            raise Exception("Invalid backend defined 'env.RL_BACKEND' = %s, Allowed backends : %s" % (
                env_backend, list(ALLOWED_BACKENDS.keys())))

        print("[rl] Using %s backend" % ALLOWED_BACKENDS[env_backend], file=sys.stderr)
        _BACKEND = env_backend
        return _BACKEND

    if _BACKEND is None:
        try:
            print("[rl] Using Torch backend", file=sys.stderr)
            _BACKEND = 'torch'
        except:
            pass

    if _BACKEND is None:
        try:
            print("[rl] Using Tensorflow backend", file=sys.stderr)
            _BACKEND = 'tf'
        except:
            pass

    if _BACKEND is None:
        raise Exception("set a backend with 'set_backend', allowed values = %s" % (ALLOWED_BACKENDS,))

    return _BACKEND


def set_seed(seed=None):
    np.random.seed(seed)
    random.seed(seed)

    if get_backend() == 'torch':
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if seed is not None:
        os.environ["RL_SEED"] = str(seed)


def get_seed():
    if "RL_SEED" in os.environ.keys():
        return int(os.environ["RL_SEED"])
    return None
