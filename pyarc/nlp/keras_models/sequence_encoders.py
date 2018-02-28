# -*- coding: utf-8 -*-

from __future__ import absolute_import

import logging
from keras.layers import Dropout, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import Conv1D, Bidirectional, LSTM, RNN
from keras.layers import Concatenate, Input, TimeDistributed
from keras.models import Model
from layers import AttentionLayer, ConsumeMask

logger = logging.getLogger(__name__)


class SequenceEncoderBase(object):
    def __init__(self, dropout_rate=0.5, **kwargs):
        ''' Create a new instance of sequence encoder.

        Args:
          dropout_rate: the final encoded output dropout rate.
        '''
        super(SequenceEncoderBase, self).__init__(**kwargs)
        self.dropout_rate = dropout_rate

    def __call__(self, inputs):
        ''' Build the actual model here.

        Args:
          inputs: the encoded or embedded sequences inputs tensor.

        Returns:
          The encoder model output tensor.
        '''
        # Avoid mask propagation when dynamic mini-batchs are not supported.
        if not self.allows_dynamic_length():
            inputs = ConsumeMask()(inputs)

        x = self.build_model(inputs)
        if self.dropout_rate > 0:
            x = Dropout(self.dropout_rate)(x)
        return x

    def build_model(self, inputs):
        ''' Build your model graph here.

        Args:
          inputs: the encoded or embedded sequences inputs tensor.

        Returns:
          The encoder model output tensor.
        '''
        raise NotImplementedError()

    def allows_dynamic_length(self):
        ''' Returns a boolean indicating whether this model is capable of
        handling variable timesteps per mini-batch.

        For example, this should be True for RNN models since you can use them
        with variable timesteps per mini-batch.
        On the other hand, CNNs expect fixed timesteps across all mini-batch.
        '''
        # Assume default as False. Should be overridden as necessary.
        return False


class HierachicalEncoder(SequenceEncoderBase):
    def __init__(self, encoders, timestep_shape, dropout_rate=0.5):
        super(HierachicalEncoder, self).__init__(dropout_rate)
        if not isinstance(encoders, (tuple, list)):
            raise ValueError('`encoders` should be of tuple/list type.')
        if not isinstance(timestep_shape, (tuple, list)):
            raise ValueError('`timestep_shape` should be of tuple/list type.')
        if not encoders or not timestep_shape:
            raise ValueError('`encoders` and `timestep_shape` size is invalid')
        if len(encoders) != len(timestep_shape):
            raise ValueError('`encoders` and `timestep_shape` should be same '
                             'size.')

        # This is required to make TimeDistributed(word_encoder_model) work.
        # TODO: Get rid of this restriction when
        # https://github.com/fchollet/keras/issues/6917 resolves.
        for ts in timestep_shape[1:]:
            if ts is None:
                raise ValueError('Values of `timestep_shape` cannot be None, '
                                 'except the first value.')

        for idx, encoder in enumerate(encoders):
            if not isinstance(encoder, SequenceEncoderBase):
                raise ValueError('%dth `encoders` should be an instance of '
                                 'SequenceEncoderBase.' % idx)
        if not encoders[0].allows_dynamic_length() \
           and timestep_shape[0] is None:
            raise ValueError('The first of `encoders` %s requires padding, '
                             'you need to provide first `timestep_shape`' %
                             encoders[0])

        self.encoders = encoders
        self.timestep_shape = timestep_shape

    def build_model(self, x):
        embedding_dim = int(x.get_shape()[-1])
        last_encoder_model = None
        for idx in range(len(self.encoders) - 1, -1, -1):
            input_shape = list(self.timestep_shape[idx:])
            input_shape.append(embedding_dim)
            last_encoding = inputs = Input(shape=input_shape)
            if last_encoder_model is not None:
                last_encoding = TimeDistributed(last_encoder_model)(inputs)
            outputs = self.encoders[idx](last_encoding)
            last_encoder_model = Model(inputs, outputs)
        res = last_encoder_model(x)
        return res


class AveragingEncoder(SequenceEncoderBase):
    ''' An encoder that averages sequence inputs. '''

    def __init__(self, dropout_rate=0):
        super(AveragingEncoder, self).__init__(dropout_rate)

    def build_model(self, x):
        return GlobalAveragePooling1D()(x)


class ShallowCNN(SequenceEncoderBase):
    ''' Yoon Kim's shallow cnn model: https://arxiv.org/pdf/1408.5882.pdf '''

    def __init__(self, filters=64, kernel_sizes=[2, 3, 4, 5], dropout_rate=0.5,
                 **conv_kwargs):
        ''' Creates a shallow CNN.

        Args:
          filters: Integer, the number of filters to use per `kernel_size`.
          kernel_sizes: An integer tuple/list, the kernel sizes of each
            convolutional layers.
          **cnn_kwargs: Additional args for building the `Conv1D` layer.
        '''
        super(ShallowCNN, self).__init__(dropout_rate)
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.conv_kwargs = conv_kwargs

    def build_model(self, x):
        pooled_tensors = []
        for kernel_size in self.kernel_sizes:
            x_i = Conv1D(self.filters, kernel_size, **self.conv_kwargs)(x)
            x_i = GlobalMaxPooling1D()(x_i)
            pooled_tensors.append(x_i)
        x = pooled_tensors[0] if len(pooled_tensors) == 0 \
            else Concatenate()(pooled_tensors)
        return x


class StackedRNN(SequenceEncoderBase):
    def __init__(self, rnn_layers=None, rnn_class=LSTM, hidden_units=[50, 50],
                 bidirectional=True, dropout_rate=0.5, **rnn_kwargs):
        ''' Creates a stacked RNN.

        Args:
          rnn_layers: List of rnn layers. If this is not None, then use this
            as inner RNNs, and ignore other parameters.
          rnn_class: The type of RNN to use.
          hidden_units: An integer tuple/list, the number of hidden units of
            RNNs.
          bidirectional: Boolean,  whether to use bidirectional encoding.
          **rnn_kwargs: Additional args for building the RNN.
        '''
        if not rnn_layers and (not rnn_class or not hidden_units):
            raise ValueError('rnn_layers and rnn_class/hidden_units cannot '
                             'both be None')

        super(StackedRNN, self).__init__(dropout_rate)
        if rnn_layers:
            for rnn in rnn_layers:
                if not isinstance(rnn, RNN):
                    raise ValueError('rnn_layers should be instances of RNN')
        self.rnn_layers = rnn_layers
        self.rnn_class = rnn_class
        self.hidden_units = hidden_units
        self.bidirectional = bidirectional
        self.rnn_kwargs = rnn_kwargs

    def build_model(self, x):
        if not self.rnn_layers:
            self.rnn_layers = []
            for i, units in enumerate(self.hidden_units):
                not_last_layer = i != len(self.hidden_units) - 1
                rnn = self.rnn_class(units, return_sequences=not_last_layer,
                                     **self.rnn_kwargs)
                if self.bidirectional:
                    rnn = Bidirectional(rnn)
                self.rnn_layers.append(rnn)
        for rnn in self.rnn_layers:
            x = rnn(x)
        return x

    def allows_dynamic_length(self):
        return True


class AttentionRNN(SequenceEncoderBase):
    def __init__(self, rnn_layer=None,
                 rnn_class=LSTM, hidden_units=50, bidirectional=True,
                 dropout_rate=0.5, **rnn_kwargs):
        ''' Creates an RNN model with attention.

        Args:
          rnn_layer: The RNN layer to use. If this is not None, then use this
            as inner RNN and ignore other parameters.
          rnn_class: The type of RNN to use.
          hidden_units: Positive integer, the number of hidden units of RNN.
          bidirectional: Boolean, whether to use bidirectional encoding.
          **rnn_kwargs: Additional args for building the RNN.
        '''
        if rnn_layer is None and rnn_class is None:
            raise ValueError('rnn_layer and rnn_class cannot both be None')
        super(AttentionRNN, self).__init__(dropout_rate)
        self.rnn_layer = rnn_layer
        self.rnn_class = rnn_class
        self.hidden_units = hidden_units
        self.bidirectional = bidirectional
        self.rnn_kwargs = rnn_kwargs

    def build_model(self, x):
        if self.rnn_layer is None:
            self.rnn_layer = self.rnn_class(self.hidden_units,
                                            return_sequences=True,
                                            **self.rnn_kwargs)
            if self.bidirectional:
                self.rnn_layer = Bidirectional(self.rnn_layer)
        token_activations = self.rnn_layer(x)
        attention_layer = AttentionLayer()
        attentted_vector = attention_layer(token_activations)
        self.attention_tensor = attention_layer.get_attention_tensor()
        return attentted_vector

    def get_attention_tensor(self):
        if not hasattr(self, 'attention_tensor'):
            raise ValueError('You need to build the model first')
        return self.attention_tensor

    def allows_dynamic_length(self):
        return True
