import sys

_BACKEND = None
allowed_backends = ('tf', 'torch',)


def set_backend(backend):
    global _BACKEND

    if backend not in allowed_backends:
        raise Exception("Allowed backends : %s", allowed_backends)

    _BACKEND = backend


def get_backend():
    global _BACKEND

    if _BACKEND is None:
        try:
            import torch as tc
            _BACKEND = 'torch'
            print("[rl] Using Torch backend", file=sys.stderr)
        except:
            pass

    if _BACKEND is None:
        try:
            import tensorflow as tf
            _BACKEND = 'tf'
            print("[rl] Using Tensorflow backend", file=sys.stderr)
        except:
            pass

    if _BACKEND is None:
        raise Exception("set a backend with 'set_backend', allowed values = %s" % (allowed_backends,))

    return _BACKEND
