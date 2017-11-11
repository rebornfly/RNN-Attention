"""
Operation functions for model.py
"""
import tensorflow as tf
import numpy as np

from tensorflow.contrib.layers.python.layers import (
    utils,
)

from tensorflow.python.framework import (
    ops,
    tensor_shape,
)

from tensorflow.python.ops import (
        gen_array_ops,
        array_ops,
        clip_ops,
        embedding_ops,
        init_ops,
        math_ops,
        nn_ops,
        partitioned_variables,
        variable_scope as vs,
)

from tensorflow.python.ops.math_ops import (
        sigmoid,
        tanh,
)

from tensorflow.python.util import (
    nest,
)

def _xavier_weight_init(nonlinearity='tanh'):
    """
    Xavier weights initialization.
    """
    def _xavier_initializer(shape, **kwargs):
        """
        Tanh and sigmoid initialization.
        """
        eps = 1.0 / np.sqrt(np.sum(shape))
        return tf.random_uniform(shape, minval=-eps, maxval=eps)

    def _relu_xavier_initializer(shape, **kwargs):
        """
        ReLU initialization.
        """
        eps = np.sqrt(2.0) / np.sqrt(np.sum(shape))
        return tf.random_uniform(shape, minval=-eps, maxval=eps)

    if nonlinearity in ('tanh', 'sigmoid'):
        return _xavier_initializer
    elif nonlinearity in ('relu'):
        return _relu_xavier_initializer
    else:
        raise Exception(
            "Please choose a valid nonlinearity: tanh|sigmoid|relu")

def _linear(args, output_size, bias, bias_start=0.0,
    nonlinearity='relu', scope=None, name=None):
    """
    Sending inputs through a two layer MLP.
    Args:
        args: list of inputs of shape (N, H)
        output_size: second dimension of W
        bias: boolean, whether or not to add bias
        bias_start: initial bias value
        nonlinearity: nonlinear transformation to use (tanh|sigmoid|relu)
        scope: (optional) Variable scope to create parameters in.
        name: (optional) variable name.
    Returns:
        Tensor with shape (N, output_size)
    """
    #print("================>",np.shape(args))
    _input = tf.concat(
        values=args,
        axis=1,)
    shape = _input.get_shape()
    #print("================>",shape)
    # Computation
    scope = vs.get_variable_scope()
    with vs.variable_scope(scope) as outer_scope:
        w_name = "W_1_"
        if name is not None:
            w_name += name
        W_1 = vs.get_variable(
            name=w_name,
            shape=[shape[1], output_size],
            initializer=_xavier_weight_init(
                nonlinearity=nonlinearity),
            )
        result_1 = tf.matmul(_input, W_1)
        if bias:
            b_name = "b_1_"
            if name is not None:
                b_name += name
            b_1 = vs.get_variable(
                name=b_name,
                shape=(output_size,),
                initializer=init_ops.constant_initializer(
                    bias_start, dtype=tf.float32),
                )
            result_1 = tf.add(result_1, b_1)
    return result_1

def ln(inputs, epsilon=1e-5, scope=None):

    """ Computer layer norm given an input tensor. We get in an input of shape
    [N X D] and with LN we compute the mean and var for each individual
    training point across all it's hidden dimensions rather than across
    the training batch as we do in BN. This gives us a mean and var of shape
    [N X 1].
    """
    mean, var = tf.nn.moments(inputs, [1], keep_dims=True)
    with tf.variable_scope(scope + 'LN'):
            scale = tf.get_variable('alpha',
                shape=[inputs.get_shape()[1]],
                initializer=tf.constant_initializer(1))
            shift = tf.get_variable('beta',
                shape=[inputs.get_shape()[1]],
                initializer=tf.constant_initializer(0))
    LN = scale * (inputs - mean) / tf.sqrt(var + epsilon) + shift

    return LN

class custom_GRUCell(tf.contrib.rnn.RNNCell):
        """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

        def __init__(self, num_units, input_size=None, activation=tanh):
            if input_size is not None:
                logging.warn("%s: The input_size parameter is deprecated.", self)
            self._num_units = num_units
            self._activation = activation

        @property
        def state_size(self):
            return self._num_units

        @property
        def output_size(self):
            return self._num_units

        def __call__(self, inputs, state, scope=None):
            """Gated recurrent unit (GRU) with nunits cells."""
            with vs.variable_scope(scope or type(self).__name__):  # "GRUCell"
                with vs.variable_scope("Gates"):  # Reset gate and update gate.
                    # We start with bias of 1.0 to not reset and not update.
                    r, u = array_ops.split(
                            _linear([inputs, state], 2 * self._num_units, True, 1.0), 2, 1,
                    )

                    # Apply Layer Normalization to the two gates
                    r = ln(r, scope = 'r/')
                    u = ln(r, scope = 'u/')

                    r, u = sigmoid(r), sigmoid(u)
                with vs.variable_scope("Candidate"):
                    c = self._activation(
                        _linear([inputs, r * state],
                            self._num_units, True))
                new_h = u * state + (1 - u) * c
            return new_h, new_h

def add_dropout_and_layers(single_cell, keep_prob, num_layers):
    """
    Add dropout and create stacked layers using a single_cell.
    """

    # Dropout
    stacked_cell = tf.contrib.rnn.DropoutWrapper(single_cell,
        output_keep_prob=keep_prob)

    # Each state as one cell
    if num_layers > 1:
        #stacked_cell = tf.nn.rnn_cell.MultiRNNCell(
        stacked_cell = tf.contrib.rnn.core_rnn_cell.MultiRNNCell(
            [single_cell] * num_layers)

    return stacked_cell
