# -*- coding: utf-8 -*-

from __future__ import absolute_import

from keras import backend as K
from keras.engine import InputSpec

from .seq2vec_encoder import Seq2VecEncoder


class BOWEncoder(Seq2VecEncoder):
    """ Bag of Words Encoder takes a tensor of shape (batch_size, num_words, word_dim)
    and returns a tensor of shape (batch_size, word_dim), which is the average
    of the (unmasked) rows in the input tensor.

    Args:
        averaged: bool. If `True`, this layer will average the embeddings
            across the time dimension, rather than summing. Default: True.
    """

    def __init__(self, averaged=True, **kwargs):
        self.averaged = averaged
        self.input_spec = [InputSpec(ndim=3)]
        super(BOWEncoder, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

    def call(self, inputs, mask=None):
        if mask is None:
            return K.mean(inputs, axis=1)
        else:
            # Compute weights such that masked elements have zero weights and the remaining
            # weight is ditributed equally among the unmasked elements.
            # Mask (samples, num_words) has 0s for masked elements and 1s everywhere else.
            # Mask is of type int8. While theano would automatically make weighted_mask below
            # of type float32 even if mask remains int8, tensorflow would complain. Let's cast it
            # explicitly to remain compatible with tf.
            float_mask = K.cast(mask, 'float32')
            # Expanding dims of the denominator to make it the same shape as the numerator, epsilon added to avoid
            # division by zero.
            # (samples, num_words)
            if self.averaged:
                weighted_mask = float_mask \
                    / (K.sum(float_mask, axis=1, keepdims=True) + K.epsilon())
            else:
                weighted_mask = float_mask
            if K.ndim(weighted_mask) < K.ndim(inputs):
                weighted_mask = K.expand_dims(weighted_mask)
            return K.sum(inputs * weighted_mask, axis=1)  # (samples, word_dim)

    def compute_mask(self, inputs, mask=None):
        # We need to override this method because Layer passes the input mask unchanged since this layer
        # supports masking. We don't want that. After the input is averaged, we can stop propagating
        # the mask.
        return None

    def get_config(self):
        config = {"averaged": self.averaged}
        base_config = super(BOWEncoder, self).get_config()
        config.update(base_config)
        return config
