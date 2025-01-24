import tensorflow as tf
from base_sequential_model import BaseSequentialModel
import numpy as np

class RNN(BaseSequentialModel):
    def __init__(self, vocab_size, max_input_len):
        """
        Initializes the RNN model with specified vocabulary size and maximum input length.

        Args:
            vocab_size (int): The size of the vocabulary, representing the total number of unique characters or tokens.
            max_input_len (int): The maximum length of input sequences that the model will process.

        This constructor also sets the model name to "RNN" and calls the
        initializer of the parent class, BaseSequentialModel, to set up
        additional model attributes.
        """
        super().__init__(vocab_size, max_input_len)
        self.model_name = "RNN"

    def set_hyperparameters(self):
        """
        Sets the hyperparameters for the RNN model.
        """
        self.hp = {
            "embedding_dim": 256,
            "rnn_units": 128,
            "learning_rate": 0.01,
            "batch_size": 128,
            "epochs": 10,
        }

    def define_model(self):
        """
        Defines a sequential RNN model for character-level text generation.

        The model consists of:
        1. An Embedding layer that converts input characters to dense vectors of size embedding_dim
        2. A SimpleRNN layer for sequence processing
        3. A Dense layer that maps to vocabulary size without activation
        4. An Activation layer with softmax to output probability distribution over vocabulary
        """
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=self.vocab_size,output_dim=self.hp["embedding_dim"],name="embedding_layer"),
            tf.keras.layers.SimpleRNN(units=self.hp["rnn_units"],name="simple_rnn_layer"),
            tf.keras.layers.Dense(units=self.vocab_size,name="dense_layer"),
            tf.keras.layers.Activation(activation='softmax',name="activation_layer")
        ])
    def define_model(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=self.vocab_size,output_dim=self.hp["embedding_dim"],name="embedding_layer"),
            tf.keras.layers.SimpleRNN(units=self.hp["rnn_units"],name="simple_rnn_layer"),
            tf.keras.layers.Dense(units=self.vocab_size,name="dense_layer"),
            tf.keras.layers.Activation(activation='softmax',name="activation_layer")
        ])

    def build_model(self):
        """
        Compiles and builds the RNN model.
        """
        learning_rate = self.hp["learning_rate"]

        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])#incldued acc metric
        self.model.build((None, self.max_input_len))  #correctde th shape