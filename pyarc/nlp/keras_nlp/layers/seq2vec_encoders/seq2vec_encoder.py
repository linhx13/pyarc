# -*- coding: utf-8 -*-

from ..masked_layer import MaskedLayer

class Seq2VecEncoder(MaskedLayer):
    """ A ``Seq2VecEncoder`` is a ``Layer`` that takes as input a sequence of
    vectors and returns a  single vector.
    Input shape: ``(batch_size, sequence_length, input_dim)``.
    Output shape: ``(batch_size, output_dim)``.
    """
    pass