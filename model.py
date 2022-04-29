import tensorflow as tf
import keras.backend as K



class AttentionLayer(tf.keras.layers.Layer):

    def __init__(self, context_vector_size=100):
        """
        Input shape: (batch_size, time_steps, input_dim)"""
        self.context_vector_size = context_vector_size
        super(AttentionLayer, self).__init__()

    def build(self, input_shape):
        dim = input_shape[2]

        # Attention layer weight
        self.W = self.add_weight(
            name="attn_weight", shape=(dim, self.context_vector_size),
            initializer=tf.keras.initializers.get("uniform"),
            trainable=True
        )

        self.U = self.add_weight(
            name="context_vector", shape=(self.context_vector_size, 1),
            initializer=tf.keras.initializers.get("uniform"),
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def _attention_weights(self, inputs, attention_mask=None):

        # u_t = K.tanh(K.dot(inputs, self.W))
        u_t = K.tanh(K.dot(inputs, self.W))
        w_u = K.dot(u_t, self.U)

        w_u = K.reshape(w_u, (-1, w_u.shape[1]))
        if attention_mask is not None:
            w_u += attention_mask
        attn_weights = K.softmax(w_u)
        return attn_weights

    def call(self, inputs, attention_mask=None, **kwargs):
        attn_weights = self._attention_weights(inputs, attention_mask)

        attn_weights = K.reshape(attn_weights, (-1, attn_weights.shape[1], 1))
        attn_weights = K.repeat_elements(attn_weights, inputs.shape[-1], -1)

        weighted_input = tf.keras.layers.Multiply()([inputs, attn_weights])
        ret = K.sum(weighted_input, axis=1)
        return ret

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]

class TextRAtt(tf.keras.Model):

    def __init__(self,
                 vocab_size=8192,
                 embed_size=128,
                 hidden_size=128,
                 context_vector_size=128,
                 num_classes=4,
                 dropout_rate=0.5,
                 use_word_dropout=True,
                 word_dropout_rate=0.3
                 ):
        super(TextRAtt, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.context_vector_size = context_vector_size
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.use_word_dropout = use_word_dropout
        self.word_dropout_rate = word_dropout_rate

        self.embeddings = tf.keras.layers.Embedding(vocab_size, embed_size, mask_zero=False)
        self.rnn = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(hidden_size, return_sequences=True)
        )
        self.han = AttentionLayer(context_vector_size)
        # self.linear = tf.keras.layers.Dense(num_classes, activation=tf.keras.layers.Softmax())
        self.linear = tf.keras.layers.Dense(num_classes, activation=None)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.word_dropout = tf.keras.layers.Dropout(word_dropout_rate)

    def call(self, inputs, training=False, **kwargs):
        embeds = self.embeddings(inputs)
        if training and self.use_word_dropout:
            # batch_size = tf.shape(embeds)[0]
            # embeds = tf.nn.dropout(embeds, self.word_dropout_rate, noise_shape=[batch_size, 1, self.embed_size])
            embeds = self.word_dropout(embeds)
            # print(embeds)

        rnn_outputs = self.rnn(embeds)
        if training:
            rnn_outputs = self.dropout(rnn_outputs)

        # Get attention mask
        mask = tf.math.equal(inputs, 0)
        mask = tf.cast(mask, dtype=tf.float32)
        attn_outputs = self.han(rnn_outputs, attention_mask=mask)

        outputs = self.linear(attn_outputs)
        return outputs

    def get_config(self):
        config = {
            "vocab_size": self.vocab_size,
            "embed_size": self.embed_size,
            "hidden_size": self.hidden_size,
            "context_vector_size": self.context_vector_size,
            "num_classes": self.num_classes,
            "dropout_rate": self.dropout_rate,
            # "base_config": super(TextHAN, self).get_config()
        }
        return config
