import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl


class CarModel(tf.keras.Model):
  class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits, **kwargs):
      # Sample a random categorical action from the given logits.
      return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

  def __init__(self, num_actions):
    super().__init__('mlp_policy')
    # Note: no tf.get_variable(), just simple Keras API!
    self.hidden1 = kl.Dense(128, activation='relu')
    self.hidden2 = kl.Dense(256, activation='relu')
    self.hidden3 = kl.Dense(128, activation='relu')
    self.value = kl.Dense(1, name='value')
    # Logits are unnormalized log probabilities.
    self.logits = kl.Dense(num_actions, activation='softmax', name='policy_logits')
    self.dist = self.ProbabilityDistribution()


  def call(self, inputs, **kwargs):
    # Inputs is a numpy array, convert to a tensor.
    #x = tf.convert_to_tensor(inputs)
    # Separate hidden layers from the same input tensor.
    hidden1_out = self.hidden1(inputs)
    hidden2_out = self.hidden2(hidden1_out)
    hidden3_out = self.hidden3(hidden2_out)
    return self.logits(hidden3_out), self.value(hidden3_out)
