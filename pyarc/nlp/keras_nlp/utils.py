# -*- coding: utf-8 -*-

import tensorflow as tf


def create_session(per_process_gpu_memory_fraction=0.5,
                   gpu_allow_growth=True,
                   allow_soft_placement=True,
                   **kwargs):
    config = tf.ConfigProto()
    config.allow_soft_placement = allow_soft_placement
    config.gpu_options.per_process_gpu_memory_fraction = \
        per_process_gpu_memory_fraction
    config.gpu_options.allow_growth = gpu_allow_growth
    session = tf.Session(config=config)
    return session
