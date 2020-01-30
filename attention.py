# -*- coding: utf-8 -*-
from keras import backend as K
from keras.engine.topology import Layer
from keras import regularizers, constraints, initializers, activations




class AttentionLayer(Layer):
    """Attention layer implementation based in the work of Yang et al. "Hierarchical
    Attention Networks for Document Classification". This implementation also allows
    changing the common tanh activation function used on the attention layer, as Chen
    et al. "A Thorough Examination of the CNN/Daily Mail Reading Comprehension Task"
    point that removing this component can be beneficial to the model. Supports
    masking.
    
    The mathematical formulation of the model is as follows:
    ```
    u = f(W * h + b),
    a_i = softmax(u_i^T * u_s),
    v_i = \sigma_i a_i * h_i.
    ```
    
    Where h are the input tensors with shape (batch, n_timesteps, hidden_size), for
    instance, all hidden vectors produced by a recurrent layer, such as a LSTM and the
    output has shape `(batch, hidden_size)`. This layer also works with inputs with more
    than 3 dimensions as well, such as sentences in a document, where each input has
    size (batch, n_docs, n_sentences, embedding_size), outputing 
    (batch, n_docs, embedding_size)`.
    
    
    # Arguments
        activation: The activation function f used by the layer (see
            [activations](../activations.md)). By default tanh is used, another common
            option is "linear".
        use_bias: Boolean, whether the layer uses a bias vector.
        initializer: Initializer for the `kernel` and `context` matrices
            (see [initializers](../initializers.md)).
        return_attention: If True, instead of returning the sequence descriptor, this
            layer will return the computed attention coefficients for each of the
            sequence timesteps. See Output section for details.
        W_regularizer: Regularizer function applied to the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        u_regularizer: Regularizer function applied to the `context` weights matrix
            (see [regularizer](../regularizers.md)).
        b_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        W_constraint: Constraint function applied to the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        u_constraint: Constraint function applied to the `contextl` weights matrix
            (see [constraints](../constraints.md)).
        b_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
    # Input shape
        nD tensor with shape: `(batch_size, ..., timesteps, input_dim)`.
        The most common situation would be a 3D input with shape
        `(batch_size, timesteps, input_dim)`.
    # Outuput shape
        The sequence descriptor with shape `(batch_size, ..., timestamps)`. If
        `return_attention` is True, this layer will return the `alpha_i` weights
        for each timestep, and consequently its output shape will be different, namely:
        `(batch_size, ..., timesteps)`.
    """
    

    def __init__(self,
                 activation='tanh',
                 initializer='glorot_uniform',
                 return_attention=False,
                 W_regularizer=None,
                 u_regularizer=None,
                 b_regularizer=None,
                 W_constraint=None,
                 u_constraint=None,
                 b_constraint=None,
                 bias=True,
                 **kwargs):
        
        self.activation = activations.get(activation)
        self.initializer = initializers.get(initializer)
        
        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        
        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)
        
        self.bias = bias
        self.supports_masking = True
        self.return_attention = return_attention

        super().__init__(**kwargs)

    def build(self, input_shape):
        # input_shape: (batch, time, amount_features)
        
        # the attention size matches the feature dimension
        amount_features = input_shape[-1]
        attention_size  = input_shape[-1]

        self.W = self.add_weight((amount_features, attention_size),
                                 initializer=self.initializer,
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint,
                                 name='attention_W')
        self.b = None
        if self.bias:
            self.b = self.add_weight((attention_size,),
                                     initializer='zero',
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint,
                                     name='attention_b')

        self.context = self.add_weight((attention_size,),
                                       initializer=self.initializer,
                                       regularizer=self.u_regularizer,
                                       constraint=self.u_constraint,
                                       name='attention_us')

        super().build(input_shape)

    def call(self, x, mask=None):        
        # U = tanh(H*W + b) (eq. 8)        
        ui = K.dot(x, self.W)              # (b, t, a)
        if self.b is not None:
            ui += self.b
        ui = self.activation(ui)           # (b, t, a)

        # Z = U * us (eq. 9)
        us = K.expand_dims(self.context)   # (a, 1)
        ui_us = K.dot(ui, us)              # (b, t, a) * (a, 1) = (b, t, 1)
        ui_us = K.squeeze(ui_us, axis=-1)  # (b, t, 1) -> (b, t)
        
        # alpha = softmax(Z) (eq. 9)
        alpha = self._masked_softmax(ui_us, mask) # (b, t)
        alpha = K.expand_dims(alpha, axis=-1)     # (b, t, 1)
        
        if self.return_attention:
            return alpha
        else:
            # v = alpha_i * x_i (eq. 10)
            return K.sum(x * alpha, axis=1)
    
    def _masked_softmax(self, logits, mask):
        """Keras's default implementation of softmax doesn't allow masking, while
        this method does if `mask` is not `None`."""
        
        # softmax(x):
        #    b = max(x)
        #    s_i = exp(xi - b) / exp(xj - b)
        #    return s
        
        b = K.max(logits, axis=-1, keepdims=True)
        logits = logits - b

        exped = K.exp(logits)

        # ignoring masked inputs
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            exped *= mask

        partition = K.sum(exped, axis=-1, keepdims=True)

        # if all timesteps are masked, the partition will be zero. To avoid this
        # issue we use the following trick:
        partition = K.maximum(partition, K.epsilon())

        return exped / partition

    def compute_output_shape(self, input_shape):
        """The attention mechanism computes a weighted average between
        all hidden vectors generated by the previous sequential layer,
        hence the input is expected to be
        `(batch_size, seq_len, amount_features)` if `return_attention` is
        `False`, otherwise the output should be (batch_size, seq_len)."""
        if self.return_attention:
            return input_shape[:-1]
        else:
            return input_shape[:-2] + input_shape[-1:]

    def compute_mask(self, x, input_mask=None):
        """This layer produces a single attended vector from a list
        of hidden vectors, hence it can't be masked as this means
        masking a single vector."""
        return None

    def get_config(self):
        config = {
            'activation': self.activation,
            'initializer': self.initializer,
            'return_attention': self.return_attention,

            'W_regularizer': initializers.serialize(self.W_regularizer),
            'u_regularizer': initializers.serialize(self.u_regularizer),
            'b_regularizer': initializers.serialize(self.b_regularizer),

            'W_constraint': constraints.serialize(self.W_constraint),
            'u_constraint': constraints.serialize(self.u_constraint),
            'b_constraint': constraints.serialize(self.b_constraint),
            
            'bias': self.bias
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
def sen_product(x, kernel):
    """
    https://github.com/richliao/textClassifier/issues/13#issuecomment-377323318
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)
    
class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    
    Note: The layer has been tested with Keras 2.0.6
    
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """
    
    def __init__(self, return_coefficients=False,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.return_coefficients = return_coefficients
        self.init = initializers.get('glorot_uniform')
        
        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        
        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)
        
        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)
    
    def build(self, input_shape):
        assert len(input_shape) == 3
        
        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        
        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)
        
        super(AttentionWithContext, self).build(input_shape)
    
    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None
    
    def call(self, x, mask=None):
        uit = sen_product(x, self.W)
        
        if self.bias:
            uit += self.b
        
        uit = K.tanh(uit)
        ait = sen_product(uit, self.u)
        
        a = K.exp(ait)
        
        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())
        
        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        
        a = K.expand_dims(a)
        weighted_input = x * a
        
        if self.return_coefficients:
            return [K.sum(weighted_input, axis=1), a]
        else:
            return K.sum(weighted_input, axis=1)
    
    
    
    def compute_output_shape(self, input_shape):
        if self.return_coefficients:
            return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[-1], 1)]
        else:
            return input_shape[0], input_shape[-1]