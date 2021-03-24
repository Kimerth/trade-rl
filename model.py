import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Dense


def countNewMean(prevMean, prevLen, newData):
    return ((prevMean * prevLen) + tf.math.reduce_sum(newData, 0)) / (prevLen + newData.shape[0])
    
def countNewStd(prevStd, prevLen, newData):
    return tf.math.sqrt(((tf.math.square(prevStd) * prevLen) + (tf.math.reduce_variance(newData, 0) * newData.shape[0])) / (prevStd + newData.shape[0]))

def normalize(data, mean = None, std = None):
    return (data - mean) / (std + 1e-8)

def build_shared_network(state_dim):
    inputs = Input(shape=state_dim)
    y = Conv1D(filters=16, kernel_size=12, strides=3, padding="same", activation="elu")   (inputs)
    y = Conv1D(filters=16, kernel_size=6, strides=2, padding="same", activation="elu")(y)
    y = Conv1D(filters=32, kernel_size=6, strides=2, padding="same", activation="elu")(y)
    y = Conv1D(filters=32, kernel_size=3, padding="same", activation="elu")(y)
    y = Flatten()(y)

    return Model(inputs=inputs, outputs=y)

def build_actor_network(state_dim, action_dim):
    inputs = Input(shape=state_dim)
    fm = build_shared_network(state_dim)(inputs)

    actor_out = Dense(action_dim, activation='softmax')(fm)
    actor = Model(inputs = inputs, outputs = actor_out)

    return actor

def build_critic_network(state_dim):
    inputs = Input(shape=state_dim)
    fm = build_shared_network(state_dim)(inputs)

    critic_out = Dense(1, activation='linear')(fm)
    critic = Model(inputs = inputs, outputs = critic_out)

    return critic

def build_RND_network(state_dim):
    inputs = Input(shape=state_dim)
    fm = build_shared_network(state_dim)(inputs)

    rnd_out = Dense(5, activation='linear')(fm)
    rnd = Model(inputs = inputs, outputs = rnd_out)

    return rnd

class ObsMemory():
    def __init__(self, state_dim):
        self.observations    = []

        self.mean_obs           = tf.zeros(state_dim, dtype = tf.float32)
        self.std_obs            = tf.zeros(state_dim, dtype = tf.float32)
        self.std_in_rewards     = tf.zeros(1, dtype = tf.float32)
        self.total_number_obs   = tf.zeros(1, dtype = tf.float32)
        self.total_number_rwd   = tf.zeros(1, dtype = tf.float32)

    def __len__(self):
        return len(self.observations)

    def get_all(self):
        return tf.constant(self.observations, dtype = tf.float32)  

    def get_all_tensor(self):
        observations = tf.constant(self.observations, dtype = tf.float32)        
        return tf.data.Dataset.from_tensor_slices(observations)

    def save_eps(self, obs):
        self.observations.append(obs)

    def save_observation_normalized(self, mean_obs, std_obs, total_number_obs):
        self.mean_obs           = mean_obs
        self.std_obs            = std_obs
        self.total_number_obs   = total_number_obs
        
    def save_rewards_normalized(self, std_in_rewards, total_number_rwd):
        self.std_in_rewards     = std_in_rewards
        self.total_number_rwd   = total_number_rwd

    def clearMemory(self):
        del self.observations[:]

from collections import namedtuple

from tensortrade.agents import ReplayMemory

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'done', 'next_state'])
class Memory(ReplayMemory):
    def __init__(self, capacity: int):
        super().__init__(capacity, Transition)

    def get_all_tensor(self):
        batch = Transition(*zip(*self.memory))

        states      = tf.convert_to_tensor(batch.state, dtype = tf.float32)
        actions     = tf.convert_to_tensor(batch.action, dtype = tf.float32)
        rewards     = tf.expand_dims(tf.convert_to_tensor(batch.reward, dtype = tf.float32), 1)
        dones       = tf.expand_dims(tf.convert_to_tensor(batch.done, dtype = tf.float32), 1)
        next_states = tf.convert_to_tensor(batch.next_state, dtype = tf.float32)
        
        return tf.data.Dataset.from_tensor_slices((states, actions, rewards, dones, next_states))   

    def clear(self):
        del self.memory[:]

def sample(datas):
    distribution = tfp.distributions.Categorical(probs = datas)
    return distribution.sample()
    
def entropy(datas):
    distribution = tfp.distributions.Categorical(probs = datas)            
    return distribution.entropy()
    
def logprob(datas, value_data):
    distribution = tfp.distributions.Categorical(probs = datas)
    return tf.expand_dims(distribution.log_prob(value_data), 1)

def kl_divergence(datas1, datas2):
    distribution1 = tfp.distributions.Categorical(probs = datas1)
    distribution2 = tfp.distributions.Categorical(probs = datas2)

    return tf.expand_dims(tfp.distributions.kl_divergence(distribution1, distribution2), 1)

class Agent():  
    def __init__(self, state_dim, action_dim, args):        
        self.args = args

        self.actor          = build_actor_network(state_dim, action_dim)
        self.actor_old      = tf.keras.models.clone_model(self.actor)

        self.ex_critic      = build_critic_network(state_dim)
        self.ex_critic_old  = tf.keras.models.clone_model(self.ex_critic)

        self.in_critic      = build_critic_network(state_dim)
        self.in_critic_old  = tf.keras.models.clone_model(self.in_critic)

        self.rnd_predict    = build_RND_network(state_dim)
        self.rnd_target     = build_RND_network(state_dim)

        self._ppo_optimizer = tf.keras.optimizers.Adam(learning_rate = args.learning_rate)
        self._rnd_optimizer = tf.keras.optimizers.Adam(learning_rate = args.learning_rate)

        self.memory         = Memory(args.mem_capacity)
        self.obs_memory     = ObsMemory(state_dim)

    def _generalized_advantage_estimation(self, values, rewards, next_values, dones):
        gae     = 0
        adv     = []     

        delta   = rewards + (1.0 - dones) * self.args.gamma * next_values - values          
        for step in reversed(range(len(rewards))):  
            gae = delta[step] + (1.0 - dones[step]) * self.args.gamma * self.args.lam * gae
            adv.insert(0, gae)
            
        return tf.stack(adv)

    def save_eps(self, state, action, reward, done, next_state):
        self.memory.push(state, action, reward, done, next_state)

    def save_observation(self, obs):
        self.obs_memory.save_eps(obs)

    def _save_obs_normalized(self, obs):
        obs                 = tf.constant(obs, dtype = tf.float32)

        mean_obs            = countNewMean(self.obs_memory.mean_obs, self.obs_memory.total_number_obs, obs)
        std_obs             = countNewStd(self.obs_memory.std_obs, self.obs_memory.total_number_obs, obs)
        total_number_obs    = len(obs) + self.obs_memory.total_number_obs
        
        self.obs_memory.save_observation_normalized(mean_obs, std_obs, total_number_obs)
    
    def _save_rwd_normalized(self, in_rewards):
        std_in_rewards      = countNewStd(self.obs_memory.std_in_rewards, self.obs_memory.total_number_rwd, in_rewards)
        total_number_rwd    = len(in_rewards) + self.obs_memory.total_number_rwd
        
        self.obs_memory.save_rewards_normalized(std_in_rewards, total_number_rwd)

    def _loss_PPO(self, action_probs, ex_values, old_action_probs, old_ex_values, next_ex_values, actions, ex_rewards, dones, 
        state_preds, state_targets, in_values, next_in_values, std_in_rewards):   

        external_advantages     = self._generalized_advantage_estimation(ex_values, ex_rewards, next_ex_values, dones)
        external_returns        = tf.stop_gradient(external_advantages + ex_values)
        external_advantages     = tf.stop_gradient((external_advantages - tf.math.reduce_mean(external_advantages)) / (tf.math.reduce_std(external_advantages) + 1e-6))

        in_rewards              = tf.math.square(state_targets - state_preds) * 0.5 / (tf.math.reduce_mean(std_in_rewards) + 1e-8)
        internal_advantages     = self._generalized_advantage_estimation(in_values, in_rewards, next_in_values, dones)
        internal_returns        = tf.stop_gradient(internal_advantages + in_values)
        internal_advantages     = tf.stop_gradient((internal_advantages - tf.math.reduce_mean(internal_advantages)) / (tf.math.reduce_std(internal_advantages) + 1e-6))

        advantages              = tf.stop_gradient(self.args.ex_advantages_coef * external_advantages + self.args.in_advantages_coef * internal_advantages)

        logprobs        = logprob(action_probs, actions)
        old_logprobs    = tf.stop_gradient(logprob(old_action_probs, actions))
        ratios          = tf.math.exp(logprobs - old_logprobs)

        kl              = kl_divergence(old_action_probs, action_probs)

        pg_loss         = tf.where(
                tf.logical_and(kl >= self.args.policy_kl_range, ratios > 1),
                ratios * advantages - self.args.policy_params * kl,
                ratios * advantages
        )
        pg_loss         = tf.math.reduce_mean(pg_loss)

        dist_entropy    = tf.math.reduce_mean(entropy(action_probs))

        ex_vf_losses1   = tf.math.square(external_returns - ex_values)
        ex_vf_losses2   = tf.math.square(external_returns - tf.stop_gradient(old_ex_values)) 
        critic_ext_loss = tf.math.reduce_mean(tf.math.maximum(ex_vf_losses1, ex_vf_losses2))  

        critic_int_loss = tf.math.reduce_mean(tf.math.square(internal_returns - in_values))

        critic_loss     = (critic_ext_loss + critic_int_loss) * 0.5

        loss            = (critic_loss * self.args.vf_loss_coef) - (dist_entropy * self.args.entropy_coef) - pg_loss
        return loss       

    @tf.function
    def act(self, state):
        state           = tf.expand_dims(tf.cast(state, dtype = tf.float32), 0)
        action_probs    = self.actor(state)
        
        if self.args.train:
            action  = sample(action_probs) 
        else:
            action  = tf.math.argmax(action_probs, 1)  
              
        return action

    @tf.function
    def _intrinsic_reward(self, obs, mean_obs, std_obs):
        obs             = normalize(obs, mean_obs, std_obs)
        
        state_pred      = self.rnd_predict(obs)
        state_target    = self.rnd_target(obs)

        return (state_target - state_pred)

    @tf.function
    def _grad_descent_rnd(self, obs, mean_obs, std_obs):
        obs             = normalize(obs, mean_obs, std_obs)
        
        with tf.GradientTape() as tape:
            state_pred      = self.rnd_predict(obs)
            state_target    = self.rnd_target(obs)

            state_target = tf.stop_gradient(state_target)        
            loss = tf.math.reduce_mean(tf.math.square(state_target - state_pred) * 0.5)

        gradients = tape.gradient(loss, self.rnd_predict.trainable_variables)        
        self._rnd_optimizer.apply_gradients(zip(gradients, self.rnd_predict.trainable_variables))

    @tf.function
    def _grad_descent_ppo(self, states, actions, rewards, dones, next_states, mean_obs, std_obs, std_in_rewards):     
        obs             = tf.stop_gradient(normalize(next_states, mean_obs, std_obs))
        state_preds     = self.rnd_predict(obs)
        state_targets   = self.rnd_target(obs)

        with tf.GradientTape() as tape:
            action_probs, ex_values, in_values  = self.actor(states), self.ex_critic(states),  self.in_critic(states)
            old_action_probs, old_ex_values     = self.actor_old(states), self.ex_critic_old(states)
            next_ex_values, next_in_values      = self.ex_critic(next_states),  self.in_critic(next_states)

            loss            = self._loss_PPO(action_probs, ex_values, old_action_probs, old_ex_values, next_ex_values, actions, rewards, dones,
                                state_preds, state_targets, in_values, next_in_values, std_in_rewards)

        gradients = tape.gradient(loss, self.actor.trainable_variables + self.ex_critic.trainable_variables + self.in_critic.trainable_variables)        
        self._ppo_optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables + self.ex_critic.trainable_variables + self.in_critic.trainable_variables)) 

    def update_rnd(self):        
        batch_size  = int(len(self.obs_memory) / self.args.minibatch)      

        intrinsic_rewards = 0
        for _ in range(self.args.RND_epochs):       
            for obs in self.obs_memory.get_all_tensor().batch(batch_size):
                self._grad_descent_rnd(obs, self.obs_memory.mean_obs, self.obs_memory.std_obs)

        intrinsic_rewards = self._intrinsic_reward(self.obs_memory.get_all(), self.obs_memory.mean_obs, self.obs_memory.std_obs)

        self._save_obs_normalized(self.obs_memory.observations)
        self._save_rwd_normalized(intrinsic_rewards)

        self.obs_memory.clearMemory()

    def update_ppo(self):        
        batch_size = int(len(self.memory) / self.args.minibatch)

        for _ in range(self.args.PPO_epochs):       
            for states, actions, rewards, dones, next_states in self.memory.get_all_tensor().batch(batch_size):
                self._grad_descent_ppo(states, actions, rewards, dones, next_states,
                    self.obs_memory.mean_obs, self.obs_memory.std_obs, self.obs_memory.std_in_rewards)

        self.memory.clear()

        self.actor_old.set_weights(self.actor.get_weights())
        self.ex_critic_old.set_weights(self.ex_critic.get_weights())
        self.in_critic_old.set_weights(self.in_critic.get_weights())

    def save_weights(self):
        self.actor.save_weights('agents/actor_ppo', save_format='tf')
        self.ex_critic.save_weights('agents/ex_critic_ppo', save_format='tf')
        self.in_critic.save_weights('agents/in_critic_ppo', save_format='tf')
        self.rnd_predict.save_weights('agents/rnd_target', save_format='tf')
        
    def load_weights(self):
        self.actor.load_weights('agents/actor_ppo')
        self.actor_old.load_weights('agents/actor_ppo')

        self.ex_critic.load_weights('agents/ex_critic_ppo')
        self.ex_critic_old.load_weights('agents/ex_critic_ppo')

        self.in_critic.load_weights('agents/in_critic_ppo')
        self.in_critic_old.load_weights('agents/in_critic_ppo')

        self.rnd_predict.load_weights('agents/rnd_target')
