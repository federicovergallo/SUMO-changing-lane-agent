import tensorflow as tf
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
import tensorflow.keras.layers as kl
import numpy as np
import logging
import datetime
import os
import random
import math
from replayMemory import ReplayMemory


def CarModel(num_actions, input_len):
    input = kl.Input(shape=(input_len))
    hidden1 = kl.Dense(128, activation='relu')(input)
    hidden2 = kl.Dense(256, activation='relu')(hidden1)
    hidden3 = kl.Dense(128, activation='relu')(hidden2)
    state_value = kl.Dense(1)(hidden3)
    state_value = kl.Lambda(lambda s: tf.keras.backend.expand_dims(s[:, 0], -1), output_shape=(num_actions,))(state_value)

    action_advantage = kl.Dense(num_actions)(hidden3)
    action_advantage = kl.Lambda(lambda a: a[:, :] - tf.keras.backend.mean(a[:, :], keepdims=True), output_shape=(num_actions,))(
        action_advantage)

    X = kl.Add()([state_value, action_advantage])

    model = tf.keras.models.Model(input, X, name='CarModel')
    return model

class DQModel(tf.keras.Model):
    def __init__(self, hidden_size=128, num_actions=3):
        super(DQModel, self).__init__()
        self.dense1 = kl.Dense(hidden_size, activation='relu')
        self.dense2 = kl.Dense(hidden_size, activation='relu')
        self.adv_dense = kl.Dense(hidden_size, activation='relu')
        self.adv_out = kl.Dense(num_actions)
        self.v_dense = kl.Dense(hidden_size, activation='relu')
        self.v_out = kl.Dense(1)
        self.lambda_layer = kl.Lambda(lambda x: x - tf.reduce_mean(x))
        self.combine = kl.Add()

    def call(self, input):
        x = self.dense1(input)
        x = self.dense2(x)
        adv = self.adv_dense(x)
        adv = self.adv_out(adv)
        v = self.v_dense(x)
        v = self.v_out(v)
        norm_adv = self.lambda_layer(adv)
        combined = self.combine([v, norm_adv])
        return combined


class DQNAgent:
    def __init__(self, fn=None, lr=0.001, gamma=0.95, batch_size=32):
        # Coefficients are used for the loss terms.
        self.gamma = gamma
        self.lr = lr
        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.checkpoint_dir = 'checkpoints/'
        self.model_name = 'DQN'
        self.model_dir = self.checkpoint_dir + self.model_name
        self.log_dir = 'logs/'
        self.train_log_dir = self.log_dir + self.model_name
        self.create_log_dir()
        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        self.fn = fn
        self.EPS_START = 0.9
        self.EPS_END = 0.5
        self.steps_done = 0
        self.EPS_DECAY = 100
        self.steps_done = 0
        self.batch_size = batch_size
        self.TAU = 0.08
        # Parameter updates
        self.loss = tf.keras.losses.Huber()
        self.optimizer = tf.optimizers.Adam(learning_rate=self.lr)
        self.main_network = CarModel(num_actions=3, input_len=37)
        self.target_network = CarModel(num_actions=3, input_len=37)

    def create_log_dir(self):
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        if not os.path.exists(self.train_log_dir):
            os.mkdir(self.train_log_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

    def act(self, state, main_network):
        # we need to do exploration vs exploitation
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.steps_done / self.EPS_DECAY)

        if np.random.rand() < eps_threshold:
            action = random.randint(0, 2)
        else:
            action = main_network.predict(np.expand_dims(state, axis=0))
            action = np.argmax(action)
        return action

    def train_step_(self, replay_memory):
        states, actions, rewards, new_states, terminal_flags = replay_memory.get_minibatch()
        q_vals = self.main_network(new_states)
        actions = np.argmax(q_vals, axis=1)
        # The target network estimates the Q-values (in the next state s', new_states is passed!)
        # for every transition in the minibatch
        q_vals = self.target_network(new_states)
        # Bellman equation. Multiplication with (1-terminal_flags) makes sure that
        # if the game is over, targetQ=rewards
        q_vals = np.array([q_vals[num, action] for num, action in enumerate(actions)])
        target_q = rewards + (self.gamma * q_vals * (1 - terminal_flags))
        loss = self.main_network.train_on_batch(states, target_q)
        return loss

    def update_network(self):
        # update target network parameters slowly from primary network
        for t, e in zip(self.main_network.trainable_variables, self.target_network.trainable_variables):
            t.assign(t * (1 - self.TAU) + e * self.TAU)

    def train(self, env, steps_per_epoch=128, epochs=10000):
        # Every four actions a gradient descend step is performed
        UPDATE_FREQ = 4
        # Number of chosen actions between updating the target network.
        NETW_UPDATE_FREQ = 10000
        # Replay mem
        REPLAY_MEMORY_START_SIZE = 33
        # Create network model
        self.main_network.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')
        # Replay memory
        my_replay_memory = ReplayMemory()
        # Metrics
        loss_avg = tf.keras.metrics.Mean()
        train_reward_tot = tf.keras.metrics.Sum()
        train_rew_comf_tot = tf.keras.metrics.Sum()
        train_rew_eff_tot = tf.keras.metrics.Sum()
        train_rew_safe_tot = tf.keras.metrics.Sum()
        train_coll_rate = tf.keras.metrics.Mean()
        train_speed_rate = tf.keras.metrics.Mean()

        # Training loop: collect samples, send to optimizer, repeat updates times.
        next_obs = env.reset(gui=True, numVehicles=40)
        first_epoch = 0
        try:
            for epoch in range(first_epoch, epochs):
                ep_rewards = 0
                for step in range(steps_per_epoch):
                    # curr state
                    state = next_obs.copy()
                    # get action
                    action = self.act(state, self.main_network)
                    # do step
                    next_obs, rewards_info, done, collision = env.step(action)
                    # process obs and get rewards
                    avg_speed_perc = env.speed / env.target_speed
                    rewards_tot, R_comf, R_eff, R_safe = rewards_info
                    # Add experience
                    my_replay_memory.add_experience(action=action,
                                                    frame=next_obs,
                                                    reward=rewards_tot,
                                                    terminal=done)
                    # Update metrics
                    train_reward_tot.update_state(rewards_tot)
                    train_rew_comf_tot.update_state(R_comf)
                    train_rew_eff_tot.update_state(R_eff)
                    train_rew_safe_tot.update_state(R_safe)
                    train_coll_rate.update_state(collision)
                    train_speed_rate.update_state(avg_speed_perc)

                    # Train every UPDATE_FREQ times
                    if self.steps_done > REPLAY_MEMORY_START_SIZE:
                        loss_value = self.train_step_(my_replay_memory)
                        loss_avg.update_state(loss_value)
                        self.update_network()
                    else:
                        loss_avg.update_state(-1)
                    # Copy network from main to target every NETW_UPDATE_FREQ
                    if step % NETW_UPDATE_FREQ == 0 and step > REPLAY_MEMORY_START_SIZE:
                        self.target_network.set_weights(self.main_network.get_weights())

                    self.steps_done += 1

                # Write
                with self.train_summary_writer.as_default():
                    tf.summary.scalar('loss', loss_avg.result(), step=epoch)
                    tf.summary.scalar('reward_tot', train_reward_tot.result(), step=epoch)
                    tf.summary.scalar('rewards_comf', train_rew_comf_tot.result(), step=epoch)
                    tf.summary.scalar('rewards_eff', train_rew_eff_tot.result(), step=epoch)
                    tf.summary.scalar('rewards_safe', train_rew_safe_tot.result(), step=epoch)
                    tf.summary.scalar('collission_rate', train_coll_rate.result(), step=epoch)
                    tf.summary.scalar('avg speed wrt maximum', train_speed_rate.result(), step=epoch)

                # Reset
                train_reward_tot.reset_states()
                train_rew_comf_tot.reset_states()
                train_rew_eff_tot.reset_states()
                train_rew_safe_tot.reset_states()
                train_coll_rate.reset_states()
                train_speed_rate.reset_states()

                # Save model
                if epoch % 1000 == 0:
                    tf.keras.models.save_model(self.main_network, self.model_dir + "/" + str(epoch) + "_main_network.hp5", save_format="h5")
                    tf.keras.models.save_model(self.target_network, self.model_dir + "/" + str(epoch) + "_target_network.hp5", save_format="h5")
        except KeyboardInterrupt:
            # self.model.save_weights(self.model_dir+"/model.ckpt")
            tf.keras.models.save_model(self.main_network, self.model_dir + "/" + str(epoch) + "_main_network.hp5", save_format="h5")
            tf.keras.models.save_model(self.target_network, self.model_dir + "/" + str(epoch) + "_target_network.hp5", save_format="h5")

        env.close()

        return 0