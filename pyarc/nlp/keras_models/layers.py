# -*- coding: utf-8 -*-

import logging

import numpy as np
from keras import backend as K
from keras.layers import Layer
from keras import initializers, regularizers, constraints

logger = logging.getLogger(__name__)


def _softmax(x, axis):
    ''' Computes softmax along a specified axis. '''
    if K.backend() == 'tensorflow':
        import tensorflow as tf
        return tf.nn.softmax(x, axis)
    elif K.backend() == 'cntk':
        import cntk
        return cntk.softmax(x, axis)
    elif K.backend() == 'theano':
        # Theano cannot softmax along an arbitrary dim.
        # So, we will shuffle `dim` to -1 and un-shuffle after softmax.
        perm = np.arange(K.ndim(x))
        perm[axis], perm[-1] = perm[-1], perm[axis]
        x_perm = K.permute_dimensions(x, perm)
        output = K.softmax(x_perm)

        # Permute back
        perm[axis], perm[-1] = perm[-1], perm[axis]
        output = K.permute_dimensions(x, output)
        return output
    else:
        raise ValueError("Backend '{}' not supported".format(K.backend()))


class AttentionLayer(Layer):
    ''' Attention layer that computes a learned attention over input sequence.

    For details, see papers:
    - https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf
    - http://colinraffel.com/publications/iclr2016feed.pdf (fig 1)


    Input:
      x: Input tensor for shape `(..., timesteps, features)` where `features`
      must be static (known).

    Output:
      2D tensor of shape `(..., features)`, i.e., `timesteps` axis is attended
      over and reduced.
    '''

    def __init__(self,
                 kernel_initializer='he_normal',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 use_bias=True,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 use_context=True,
                 context_initializer='he_normal',
                 context_regularizer=None,
                 context_constraint=None,
                 attention_dim=None,
                 **kwargs):
        '''
        Args:
          attention_dim: The dimensionality of the inner attention calculating
            neural network.
            For input `(32, 10, 300)`, with `attention_dim` of 100, the output
            is `(32, 10, 100)`, i.e., the attended words are 100 dimensional.
            This is then collapsed via summation to `(32, 10, 1)` to indicate
            the attention weights for 10 words.
            If set to None, `features` dims are used as `attention_dim`.
            (Default value: None)
        '''
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(AttentionLayer, self).__init__(**kwargs)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)

        self.use_bias = use_bias
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)

        self.use_context = use_context
        self.context_initializer = initializers.get(context_initializer)
        self.context_regularizer = regularizers.get(context_regularizer)
        self.context_constraint = constraints.get(context_constraint)

        self.attention_dim = attention_dim
        self.supports_masking = True

    def build(self, input_shape):
        # if len(input_shape) < 3:
        if len(input_shape) != 3:
            raise ValueError("Expected input shape of "
                             "`(..., timesteps, features)`, found `{}`"
                             .format(input_shape))

        attention_dim = input_shape[-1] if self.attention_dim is None \
            else self.attention_dim
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[-1], attention_dim),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=(attention_dim,),
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        if self.use_context:
            self.context_kernel = self.add_weight(
                name='context_kernel',
                shape=(attention_dim,),
                initializer=self.context_initializer,
                regularizer=self.context_regularizer,
                constraint=self.context_constraint)
        else:
            self.context_kernel = None
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs, mask=None):
        # inputs: [..., timesteps, features]
        # ut : [..., timesteps, attention_dim]
        ut = K.dot(inputs, self.kernel)
        if self.use_bias:
            ut = K.bias_add(ut, self.bias)
        ut = K.tanh(ut)
        if self.use_context:
            ut = ut * self.context_kernel

        # Collapse `attention_dim` to 1. This indicates the weight for each
        # timestep.
        ut = K.sum(ut, axis=-1, keepdims=True)
        # ut = K.sum(ut, axis=-1)

        # Convert those weights into a distribution along timestep axis.
        # i.e., sum of alphas along `timesteps` axis should be 1.
        self.at = _softmax(ut, axis=1)
        # self.at = K.expand_dims(K.softmax(ut), -1)
        if mask is not None:
            self.at *= K.cast(K.expand_dims(mask, -1), K.floatx())

        # Weighted sum along `timesteps` axis.
        return K.sum(inputs * self.at, axis=-2)

    def comput_mask(self, inputs, input_mask=None):
        # Do not pass the mask to the next layers
        return None

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]
        # input_shape = list(input_shape)
        # input_shape[-2] = input_shape[-1]
        # return tuple(input_shape[:-1])

    def get_attention_tensor(self):
        if not hasattr(self, 'at'):
            raise ValueError('Attention tensor is available after calling this'
                             ' layer with an input')
        return self.at

        def get_config(self):
            config = {
                'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
                'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
                'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
                'bias_initializer':
                initializers.serialize(self.bias_initializer),
                'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
                'bias_constraint':
                constraints.serialize(self.bias_constraint),
                'context_initializer':
                initializers.serialize(self.context_initializer),
                'context_regularizer':
                regularizers.serialize(self.context_regularizer),
                'context_constraint':
                constraints.serialize(self.context_constraint)
            }
            base_config = super(AttentionLayer, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))


class ConsumeMask(Layer):
    ''' Layer that prevents mask propagation. '''

    def compute_mask(self, inputs, input_mask=None):
        # Do not pass the mask to the next layer.
        return None

    def call(self, inputs, mask=None):
        return inputs
