_SESS_CONFIG = None

import tensorflow as tf


def set_session_config(config):
    global _SESS_CONFIG
    _SESS_CONFIG = config


def _sess_config():
    global _SESS_CONFIG
    if _SESS_CONFIG is None:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        _SESS_CONFIG = config
    return _SESS_CONFIG
