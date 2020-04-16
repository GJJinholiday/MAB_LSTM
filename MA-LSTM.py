# -*- coding: utf-8 -*
from keras.layers import Layer
import keras.backend as K
import tensorflow as tf

def cumsoftmax(x, mode='l2r'):
    axis = K.ndim(x) - 1
    if mode == 'l2r':
        x = K.softmax(x, axis=axis)
        x = K.cumsum(x, axis=axis)
        return x
    elif mode == 'r2l':
        x = x[..., ::-1]
        x = K.softmax(x, axis=axis)
        x = K.cumsum(x, axis=axis)
        return x[..., ::-1]
    else:
        return x


class MA_LSTM(Layer):
    def __init__(self,
                 units,
                 levels,
                 return_sequences=False,
                 dropconnect=None,
                 dk=1024,
                 **kwargs):
        assert units % levels == 0
        self.units = units
        self.return_sequences = return_sequences
        self.dk=dk
        self.dv=dk
        self.dropconnect = dropconnect
        super(ONLSTM2, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 5),
            name='kernel',
            initializer='glorot_uniform')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 5),
            name='recurrent_kernel',
            initializer='orthogonal')
        self.cell_kernel = self.add_weight(
            shape=(self.units, self.units * 5),
            name='cell_kernel',
            initializer='orthogonal')
        self.up_att_downl = self.add_weight(
            shape=(1, self.dk),
            name='up_att_downl',
            initializer='glorot_uniform'
            )
        self.up_att_downr = self.add_weight(
            shape=(1, self.dk),
            name='up_att_downr',
            initializer='glorot_uniform'
        )
        self.up_att_randl = self.add_weight(
            shape=(1, self.dk),
            name='up_att_randl',
            initializer='glorot_uniform'
        )
        self.up_att_randr = self.add_weight(
            shape=(1, self.dk),
            name='up_att_randr',
            initializer='glorot_uniform'
        )
        self.down_att_upl = self.add_weight(
            shape=(1, self.dk),
            name='down_att_upl',
            initializer='glorot_uniform'
        )
        self.down_att_upr = self.add_weight(
            shape=(1, self.dk),
            name='down_att_upr',
            initializer='glorot_uniform'
        )
        self.down_att_randl = self.add_weight(
            shape=(1, self.dk),
            name='down_att_randl',
            initializer='glorot_uniform'
        )
        self.down_att_randr = self.add_weight(
            shape=(1, self.dk),
            name='down_att_randr',
            initializer='glorot_uniform'
        )
        self.rand_att_upl = self.add_weight(
            shape=(1, self.dk),
            name='rand_att_upl',
            initializer='glorot_uniform'
        )
        self.rand_att_upr = self.add_weight(
            shape=(1, self.dk),
            name='rand_att_upr',
            initializer='glorot_uniform'
        )
        self.rand_att_downl = self.add_weight(
            shape=(1, self.dk),
            name='rand_att_downl',
            initializer='glorot_uniform'
        )
        self.rand_att_downr = self.add_weight(
            shape=(1, self.dk),
            name='rand_att_downr',
            initializer='glorot_uniform'
        )
        self.aggregation = self.add_weight(
            shape=(self.units * 3, self.units),
            name = 'aggregation',
            initializer='glorot_uniform'
        )
        self.bias = self.add_weight(
            shape=(self.units * 5,),
            name='bias',
            initializer='zeros')
        self.built = True
        if self.dropconnect:
            self._kernel = K.dropout(self.kernel, self.dropconnect)
            self._kernel = K.in_train_phase(self._kernel, self.kernel)
            self._recurrent_kernel = K.dropout(self.recurrent_kernel, self.dropconnect)
            self._recurrent_kernel = K.in_train_phase(self._recurrent_kernel, self.recurrent_kernel)
            self._cell_kernel = K.dropout(self.cell_kernel, self.dropconnect)
            self._cell_kernel = K.in_train_phase(self._cell_kernel, self.cell_kernel)
        else:
            self._kernel = self.kernel
            self._recurrent_kernel = self.recurrent_kernel
            self._cell_kernel = self.cell_kernel

    def one_step(self, inputs, states):
        x_in, (c_last, h_last) = inputs, states
        x_out = K.dot(x_in, self._kernel) + K.dot(h_last, self._recurrent_kernel) + K.dot(c_last, self._cell_kernel)
        x_out = K.bias_add(x_out, self.bias)
        up = cumsoftmax(x_out[:, :self.units], 'l2r')
        up = K.expand_dims(up, 2)
        down = cumsoftmax(x_out[:, self.units: self.units * 2], 'r2l')
        down = K.expand_dims(down, 2)
        rand = x_out[:, self.units * 2: self.units * 3]
        rand = K.expand_dims(rand, 2)
        O11 = K.sigmoid(K.batch_dot(K.batch_dot(K.dot(up, self.up_att_downl), tf.transpose(K.dot(down, self.up_att_downr), perm=[0,2,1])), up))
        O12 = K.sigmoid(K.batch_dot(K.batch_dot(K.dot(up, self.up_att_randl), tf.transpose(K.dot(rand, self.up_att_randr), perm=[0,2,1])), up))
        O1 = K.squeeze(O11 + O12, axis=-1)
        O21 = K.sigmoid(K.batch_dot(K.batch_dot(K.dot(rand, self.rand_att_upl), tf.transpose(K.dot(up, self.rand_att_upr), perm=[0,2,1])), rand))
        O22 = K.sigmoid(K.batch_dot(K.batch_dot(K.dot(rand, self.rand_att_downl), tf.transpose(K.dot(down, self.rand_att_downr), perm=[0,2,1])), rand))
        O2 = K.squeeze(O21 + O22, axis=-1)
        O31 = K.sigmoid(K.batch_dot(K.batch_dot(K.dot(down, self.down_att_upl), tf.transpose(K.dot(up, self.down_att_upr), perm=[0,2,1])), down))
        O32 = K.sigmoid(K.batch_dot(K.batch_dot(K.dot(down, self.down_att_randl), tf.transpose(K.dot(rand, self.down_att_randr), perm=[0,2,1])), down))
        O3 = K.squeeze(O31 + O32, axis=-1)
        O = K.dot(K.concatenate([O1,O2,O3],axis=-1),  self.aggregation)
        f_gate = K.sigmoid(O)
        i_gate = 1 - f_gate
        o_gate = x_out[:, self.units * 3: self.units * 4]
        c_in = K.tanh(x_out[:, self.units * 4:])
        c_out = f_gate * c_last + i_gate * c_in
        h_out = o_gate * K.tanh(c_out)
        out = K.concatenate([h_out, f_gate, i_gate], 1)
        return out, [c_out, h_out]

    def call(self, inputs):
        initial_states = [
            K.zeros((K.shape(inputs)[0], self.units)),
            K.zeros((K.shape(inputs)[0], self.units))
        ] # 定义初始态(全零)
        outputs = K.rnn(self.one_step, inputs, initial_states)
        self.distance = 1 - K.mean(outputs[1][..., self.units: self.units * 2], -1)
        self.distance_in = K.mean(outputs[1][..., self.units * 2:], -1)
        if self.return_sequences:
            return outputs[1][..., :self.units]
        else:
            return outputs[0][..., :self.units]

    def compute_output_shape(self, input_shape):
        if self.return_sequences:
            return (input_shape[0], input_shape[1], self.units)
        else:
            return (input_shape[0], self.units)
