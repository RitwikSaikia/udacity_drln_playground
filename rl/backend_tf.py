_TF_SESSION = None

import tensorflow as tf

def set_session(sess):
    global _TF_SESSION
    _TF_SESSION = sess


def _sess():
    global _TF_SESSION
    if _TF_SESSION is None:
        _TF_SESSION = tf.Session()
    return _TF_SESSION
