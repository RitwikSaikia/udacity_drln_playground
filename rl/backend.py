import os
import sys

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
