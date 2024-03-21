import tensorflow as tf

from tcn_sequence_models.tf_models.tcn import TCN
# from tensorflow.keras.layers import MultiHeadAttention


class Encoder(tf.keras.Model):
    def __init__(
        # self,
        # max_seq_len: int,
        # num_filters: int,
        # kernel_size: int,
        # dilation_base: int,
        # dropout_rate: float,
        # num_layers: int = None,
        # activation: str = "elu",
        # kernel_initializer: str = "he_normal",
        # batch_norm: bool = False,
        # layer_norm: bool = False,
        # padding: str = "causal",
        # key_size: int,
        # value_size: int,
        # num_attention_heads: int,
        self,
        max_seq_len: int,
        num_filters: int,
        kernel_size: int,
        dilation_base: int,
        dropout_rate: float,
        # key_size: int,
        # value_size: int,
        # num_attention_heads: int,
        output_neurons: [int],
        num_layers: int,
        activation: str = "elu",
        kernel_initializer: str = "he_normal",
        batch_norm: bool = False,
        layer_norm: bool = False,
        # autoregressive: bool = False,
        padding: str = "causal",
    ):
        """TCN Encoder stage
        The encoder consists of a TCN block.

        :param max_seq_len: maximum sequence length that is used to compute the
        number of layers
        :param num_filters: number of channels for CNNs
        :param kernel_size: kernel size of CNNs
        :param dilation_base: dilation base
        :param dropout_rate: dropout rate
        :param num_layers: number of layer in the TCNs. If None, the needed
        number of layers is computed automatically based on the sequence lenghts
        :param activation: the activation function used throughout the encoder
        :param kernel_initializer: the mode how the kernels are initialized
        :param batch_norm: if batch normalization shall be used
        :param layer_norm: if layer normalization shall be used
        :param padding: Padding mode of the encoder. One of ['causal', 'same']
        """
        super(Encoder, self).__init__()
        # self.max_seq_len = max_seq_len
        # self.num_filters = num_filters
        # self.kernel_size = kernel_size
        # self.dilation_base = dilation_base
        # self.dropout_rate = dropout_rate
        # self.num_layers = num_layers
        # self.activation = activation
        # self.kernel_initializer = kernel_initializer
        # self.batch_norm = batch_norm
        # self.layer_norm = layer_norm
        # self.padding = padding
        # self.key_size = key_size
        # self.value_size = value_size
        # self.num_attention_heads = num_attention_heads
        self.max_seq_len = max_seq_len
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dilation_base = dilation_base
        self.dropout_rate = dropout_rate
        # self.key_size = key_size
        # self.value_size = value_size
        # self.num_attention_heads = num_attention_heads
        self.output_neurons = output_neurons
        self.num_layers = num_layers
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        # self.autoregressive = autoregressive
        self.padding = padding 

        self.tcn = TCN(
            max_seq_len=self.max_seq_len,
            num_stages=2,
            num_filters=self.num_filters,
            kernel_size=self.kernel_size,
            dilation_base=self.dilation_base,
            dropout_rate=self.dropout_rate,
            activation=self.activation,
            final_activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            padding=self.padding,
            weight_norm=False,
            batch_norm=self.batch_norm,
            layer_norm=self.layer_norm,
            num_layers=self.num_layers,
            return_sequence=True,
        )

     #Cross attention
        # self.attention = MultiHeadAttention(
        #     key_dim=self.key_size,
        #     value_dim=self.value_size,
        #     num_heads=self.num_attention_heads,
        #     output_shape=self.num_filters,
        # )

        # if self.layer_norm:
        #     self.normalization_layer = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # else:
        #     self.normalization_layer = tf.keras.layers.BatchNormalization()

    @tf.function
    def call(self, data_encoder, training=None):
        return self.tcn(data_encoder, training=training)
        # out_attention = self.attention(out_tcn, data_encoder, training=True)
        # out = self.normalization_layer(out_tcn + out_attention, training=True)


