from model import Agent
from tensortrade.env.generic import TradingEnv
import argparse

def parse_args(args):
    parser = argparse.ArgumentParser(description='Training parameters')

    parser.add_argument('--load_weights', type=bool, default=False)
    parser.add_argument('--save_weights', type=bool, default=True)
    parser.add_argument('--train', type=bool, default=True)

    parser.add_argument('--n_step_update', type=int, default=96)
    parser.add_argument('--n_eps_update', type=int, default=1)
    parser.add_argument('--nb_episodes', type=int, default=100)
    parser.add_argument('--n_init_episode', type=int, default=1024)
    parser.add_argument('--n_saved', type=int, default=2)


    parser.add_argument('--policy_kl_range', type=float, default=0.03)
    parser.add_argument('--policy_params', type=int, default=20)
    parser.add_argument('--entropy_coef', type=float, default=0.05)
    parser.add_argument('--vf_loss_coef', type=float, default=1.0)
    parser.add_argument('--minibatch', type=int, default=64)
    parser.add_argument('--PPO_epochs', type=int, default=10)
    parser.add_argument('--RND_epochs', type=int, default=5)
    parser.add_argument('--mem_capacity', type=int, default=10000)

    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--learning_rate', type=float, default=2e-43)

    parser.add_argument('--ex_advantages_coef', type=int, default=2)
    parser.add_argument('--in_advantages_coef', type=int, default=1)

    #parser.add_argument('--consecutive_frames', type=int, default=4, help="Number of consecutive frames (action repeat)")

    parser.add_argument('--render', dest='render', action='store_true')

    parser.set_defaults(render=False)
    return parser.parse_args(args)

def run_inits_episode(env: TradingEnv, agent: Agent, n_init_episode: int):
    env.reset()

    for _ in range(n_init_episode):
        action                  = env.action_space.sample()
        next_state, _, done, _  = env.step(action)
        agent.save_observation(next_state)

        if done:
            env.reset()

    agent._save_obs_normalized(agent.obs_memory.observations)
    agent.obs_memory.clearMemory()

    return agent

def run_episode(env: TradingEnv, agent: Agent, training_mode: bool, t_updates: int, n_update: int, episode: int):
    state           = env.reset()
    done            = False
    total_reward    = 0
    eps_time        = 0
    
    while not done:
        action                      = int(agent.act(state))
        next_state, reward, done, _ = env.step(action)
        
        eps_time        += 1 
        t_updates       += 1
        total_reward    += reward

        if training_mode:
            agent.save_eps(state.tolist(), float(action), float(reward), float(done), next_state.tolist())
            agent.save_observation(next_state)
            
        state   = next_state
        
        if training_mode:
            if t_updates % n_update == 0:
                agent.update_rnd()
                t_updates = 0
            
            if eps_time % 1000 == 0:
                env.render(episode=episode, step=eps_time)

            if eps_time > 5000:
                done = True
        
        if done:           
            return total_reward, eps_time, t_updates           

import random
import time
from env import get_train_env
import sys

def str_time_prop(start, end, format, prop):
    stime = time.mktime(time.strptime(start, format))
    etime = time.mktime(time.strptime(end, format))

    ptime = stime + prop * (etime - stime)

    return time.strftime(format, time.localtime(ptime))


def random_date(start, end, prop):
    return str_time_prop(start, end, '%Y-%m-%d %H:%M:%S', prop)

def main(args = None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    env                 = get_train_env()

    state_dim           = env.observation_space.shape
    action_dim          = env.action_space.n

    agent               = Agent(state_dim, action_dim, args)  

    if args.load_weights:
        agent.load_weights()
        print('Weight Loaded')

    t_updates           = 0

    if args.train:
        agent = run_inits_episode(env, agent, args.n_init_episode)

    for i_episode in range(1, args.nb_episodes + 1):
        env = get_train_env(random_date("2018-01-01 00:00:00", "2021-02-01 00:00:00", random.random()))
        total_reward, time, t_updates = run_episode(env, agent, args.train, t_updates, args.n_step_update, i_episode)
        print('Episode {} \t t_reward: {} \t time: {} \t '.format(i_episode, total_reward, time))

        if i_episode % args.n_eps_update == 0:
            agent.update_ppo()       

        if args.save_weights:
            if i_episode % args.n_saved == 0:
                agent.save_weights() 
                print('weights saved')

    print('========== Final ==========')

if __name__ == '__main__':
    main()