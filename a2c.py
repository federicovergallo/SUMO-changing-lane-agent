import tensorflow as tf
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
import tensorflow.keras.layers as kl
import numpy as np
import logging
import datetime
import os


def ProbabilityDistribution():
    inputs = kl.Input(shape=(3))
    sample = tf.random.categorical(inputs, 1)
    squeeze = tf.squeeze(sample, axis=-1)
    model = tf.keras.models.Model(inputs, squeeze, name='dist')
    return model


def CarModel(num_actions, input_len):
    input = kl.Input(shape=(input_len))
    hidden1 = kl.Dense(128, activation='relu')(input)
    hidden2 = kl.Dense(256, activation='relu')(hidden1)
    hidden3 = kl.Dense(128, activation='relu')(hidden2)
    value = kl.Dense(1, name='value')(hidden3)
    # Logits are unnormalized log probabilities.
    logits = kl.Dense(num_actions, activation='softmax', name='policy_logits')(hidden3)

    model = tf.keras.models.Model(input, [logits, value], name='CarModel')
    return model

class A2CAgent:
  def __init__(self, fn=None, lr=0.0005, gamma=0.95, value_c=0.5, entropy_c=1e-4):
    # Coefficients are used for the loss terms.
    self.value_c = value_c
    self.entropy_c = entropy_c
    self.gamma = gamma
    self.lr = lr
    self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    self.checkpoint_dir = 'checkpoints/'
    self.model_dir = self.checkpoint_dir + self.current_time
    self.log_dir = 'logs/'
    self.train_log_dir = self.log_dir + self.current_time
    self.create_log_dir()
    self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
    self.fn = fn

  def test(self, env, render=True):
    obs, done, ep_reward = env.reset(), False, 0
    while not done:
      action, _ = self.action_value(self.model, obs[None, :])
      obs, reward, done = env.step(action)
      ep_reward += reward
      if render:
        env.render()
    return ep_reward

  def create_log_dir(self):
      if not os.path.exists(self.log_dir):
          os.mkdir(self.log_dir)
      if not os.path.exists(self.train_log_dir):
          os.mkdir(self.train_log_dir)
      if not os.path.exists(self.checkpoint_dir):
          os.mkdir(self.checkpoint_dir)
      if not os.path.exists(self.model_dir):
          os.mkdir(self.model_dir)

  def train(self, env, steps_per_epoch=64, epochs=50000):
      # Create network model
      self.dist = ProbabilityDistribution()
      self.model = CarModel(num_actions=3, input_len=env.state_dim)
      if self.fn is not None:
          # self.model.load_weights(fn)
          self.model = tf.keras.models.load_model(self.fn, custom_objects={'_logits_loss': self._logits_loss,
                                                                      '_value_loss': self._value_loss,
                                                                      'action_value': self.action_value})
      self.model.compile(
          optimizer=ko.Adam(lr=self.lr),
          # Define separate losses for policy logits and value estimate.
          loss=[self._logits_loss, self._value_loss])

      # Storage helpers for a single batch of data.
      actions = np.empty((steps_per_epoch,), dtype=np.int32)
      rewards_tot, R_comf, R_eff, R_safe, dones, values, collisions, avg_speed_perc = np.empty((8, steps_per_epoch))
      observations = np.empty((steps_per_epoch,env.state_dim))

      # Training loop: collect samples, send to optimizer, repeat updates times.
      ep_rewards = [0.0]
      next_obs = env.reset(gui=True, numVehicles=35)
      first_epoch = 0
      try:
          for epoch in range(first_epoch, epochs):
              for step in range(steps_per_epoch):
                  observations[step] = next_obs.copy()
                  actions[step], values[step] = self.action_value(self.model, next_obs[None, :])
                  next_obs, rewards_info, dones[step], collision = env.step(actions[step])
                  avg_speed_perc[step] = env.speed/env.target_speed
                  rewards_tot[step], R_comf[step], R_eff[step], R_safe[step] = rewards_info
                  collisions[step] = collision
                  ep_rewards[-1] += rewards_tot[step]

              _, next_value = self.action_value(self.model, next_obs[None, :])

              returns, advs = self._returns_advantages(rewards_tot, dones, values, next_value)
              # A trick to input actions and advantages through same API.
              acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)

              # Performs a full training step on the collected batch.
              # Note: no need to mess around with gradients, Keras API handles it.
              losses = self.model.train_on_batch(observations, [acts_and_advs, returns])

              logging.info("[%d/%d] Losses: %s" % (epoch + 1, epochs, losses))

              with self.train_summary_writer.as_default():
                  tf.summary.scalar('loss', np.mean(losses), step=epoch)
                  tf.summary.scalar('reward_tot', np.mean(rewards_tot), step=epoch)
                  tf.summary.scalar('rewards_comf', np.sum(R_comf), step=epoch)
                  tf.summary.scalar('rewards_eff', np.sum(R_eff), step=epoch)
                  tf.summary.scalar('rewards_safe', np.sum(R_safe), step=epoch)
                  tf.summary.scalar('collission_rate', np.mean(collisions), step=epoch)
                  tf.summary.scalar('avg speed wrt maximum', np.mean(avg_speed_perc), step=epoch)

              if epoch % 100 == 0:
                  tf.keras.models.save_model(self.model, self.model_dir+"/model.hp5", save_format="h5")

      except KeyboardInterrupt:
          #self.model.save_weights(self.model_dir+"/model.ckpt")
          tf.keras.models.save_model(self.model, self.model_dir + "/model.hp5", save_format="h5")

      env.close()

      return ep_rewards

  def _returns_advantages(self, rewards, dones, values, next_value):
      # `next_value` is the bootstrap value estimate of the future state (critic).
      returns = np.append(np.zeros_like(rewards), next_value)

      # Returns are calculated as discounted sum of future rewards.
      for t in reversed(range(rewards.shape[0])):
          returns[t] = rewards[t] + self.gamma * returns[t + 1] * (1 - dones[t])
      returns = returns[:-1]

      # Advantages are equal to returns - baseline (value estimates in our case).
      advantages = returns - values

      return returns, advantages

  def _value_loss(self, returns, value):
      # Value loss is typically MSE between value estimates and returns.
      return self.value_c * kls.mean_squared_error(returns, value)

  def _logits_loss(self, actions_and_advantages, logits):
      # A trick to input actions and advantages through the same API.
      actions, advantages = tf.split(actions_and_advantages, 2, axis=-1)

      # Sparse categorical CE loss obj that supports sample_weight arg on `call()`.
      # `from_logits` argument ensures transformation into normalized probabilities.
      weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)

      # Policy loss is defined by policy gradients, weighted by advantages.
      # Note: we only calculate the loss on the actions we've actually taken.
      actions = tf.cast(actions, tf.int32)
      policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)

      # Entropy loss can be calculated as cross-entropy over itself.
      probs = tf.nn.softmax(logits)
      entropy_loss = kls.categorical_crossentropy(probs, probs)

      # We want to minimize policy and maximize entropy losses.
      # Here signs are flipped because the optimizer minimizes.
      return policy_loss - self.entropy_c * entropy_loss


  def action_value(self, model, obs):
    # Executes `call()` under the hood.
    logits, value = model.predict_on_batch(obs)
    action = self.dist.predict_on_batch(logits)
    # Another way to sample actions:
    #   action = tf.random.categorical(logits, 1)
    # Will become clearer later why we don't use it.
    return action, np.squeeze(value)