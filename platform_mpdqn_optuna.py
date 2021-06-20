import os
import click
import time
import gym
import gym_platform
from gym.wrappers import Monitor
from common import ClickPythonLiteralOption
from common.platform_domain import PlatformFlattenedActionWrapper

import numpy as np
import optuna 
from common.wrappers import ScaledStateWrapper, ScaledParameterisedActionWrapper


def pad_action(act, act_param):
    params = [np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32)]
    params[act][:] = act_param
    return (act, params)


def evaluate(env, agent, episodes=1000):
    returns = []
    timesteps = []
    for _ in range(episodes):
        state, _ = env.reset()
        terminal = False
        t = 0
        total_reward = 0.
        while not terminal:
            t += 1
            state = np.array(state, dtype=np.float32, copy=False)
            act, act_param, all_action_parameters = agent.act(state)
            action = pad_action(act, act_param)
            (state, _), reward, terminal, _ = env.step(action)
            total_reward += reward
        timesteps.append(t)
        returns.append(total_reward)
    # return np.column_stack((returns, timesteps))
    return np.array(returns)


def run(seed, episodes, evaluation_episodes, batch_size, gamma, inverting_gradients, initial_memory_threshold,
        replay_memory_size, epsilon_steps, tau_actor, tau_actor_param, use_ornstein_noise, learning_rate_actor,
        learning_rate_actor_param, epsilon_final, zero_index_gradients, initialise_params, scale_actions,
        clip_grad, split, indexed, layers, multipass, weighted, average, random_weighted, render_freq,
        save_freq, save_dir, save_frames, visualise, action_input_layer, title):

  #  if save_freq > 0 and save_dir:
  #      save_dir = os.path.join(save_dir, title + "{}".format(str(seed)))
  #      os.makedirs(save_dir, exist_ok=True)
  #  assert not (save_frames and visualise)
  #  if visualise:
  #      assert render_freq > 0
  #  if save_frames:
  #      assert render_freq > 0
  #      vidir = os.path.join(save_dir, "frames")
  #      os.makedirs(vidir, exist_ok=True)

    env = gym.make('Platform-v0')
    initial_params_ = [3., 10., 400.]
    if scale_actions:
        for a in range(env.action_space.spaces[0].n):
            initial_params_[a] = 2. * (initial_params_[a] - env.action_space.spaces[1].spaces[a].low) / (
                        env.action_space.spaces[1].spaces[a].high - env.action_space.spaces[1].spaces[a].low) - 1.

    env = ScaledStateWrapper(env)
    env = PlatformFlattenedActionWrapper(env)
    if scale_actions:
        env = ScaledParameterisedActionWrapper(env)

    dir = os.path.join(save_dir,title)
    env = Monitor(env, directory=os.path.join(dir,str(seed)), video_callable=False, write_upon_reset=False, force=True)
    env.seed(seed)
    np.random.seed(seed)

    print(env.observation_space)

    from agents.pdqn import PDQNAgent
    from agents.pdqn_split import SplitPDQNAgent
    from agents.pdqn_multipass import MultiPassPDQNAgent
    
    agent_class = MultiPassPDQNAgent
    agent = agent_class(
                       env.observation_space.spaces[0], env.action_space,
                       batch_size=batch_size,
                       learning_rate_actor=learning_rate_actor,
                       learning_rate_actor_param=learning_rate_actor_param,
                       epsilon_steps=epsilon_steps,
                       gamma=gamma,
                       tau_actor=tau_actor,
                       tau_actor_param=tau_actor_param,
                       clip_grad=clip_grad,
                       indexed=indexed,
                       weighted=weighted,
                       average=average,
                       random_weighted=random_weighted,
                       initial_memory_threshold=initial_memory_threshold,
                       use_ornstein_noise=use_ornstein_noise,
                       replay_memory_size=replay_memory_size,
                       epsilon_final=epsilon_final,
                       inverting_gradients=inverting_gradients,
                       actor_kwargs={'hidden_layers': layers,
                                     'action_input_layer': action_input_layer,},
                       actor_param_kwargs={'hidden_layers': layers,
                                           'squashing_function': False,
                                           'output_layer_init_std': 0.0001,},
                       zero_index_gradients=zero_index_gradients,
                       seed=seed)

    if initialise_params:
        initial_weights = np.zeros((env.action_space.spaces[0].n, env.observation_space.spaces[0].shape[0]))
        initial_bias = np.zeros(env.action_space.spaces[0].n)
        for a in range(env.action_space.spaces[0].n):
            initial_bias[a] = initial_params_[a]
        agent.set_action_parameter_passthrough_weights(initial_weights, initial_bias)
    print(agent)
    max_steps = 250
    total_reward = 0.
    returns = []
    start_time = time.time()
    video_index = 0
    # agent.epsilon_final = 0.
    # agent.epsilon = 0.
    # agent.noise = None

    for i in range(episodes):
        if save_freq > 0 and save_dir and i % save_freq == 0:
            agent.save_models(os.path.join(save_dir, str(i)))
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32, copy=False)
        if visualise and i % render_freq == 0:
            env.render()

        act, act_param, all_action_parameters = agent.act(state)
        action = pad_action(act, act_param)

        episode_reward = 0.
        agent.start_episode()
        for j in range(max_steps):

            ret = env.step(action)
            (next_state, steps), reward, terminal, _ = ret
            next_state = np.array(next_state, dtype=np.float32, copy=False)

            next_act, next_act_param, next_all_action_parameters = agent.act(next_state)
            next_action = pad_action(next_act, next_act_param)
            agent.step(state, (act, all_action_parameters), reward, next_state,
                       (next_act, next_all_action_parameters), terminal, steps)
            act, act_param, all_action_parameters = next_act, next_act_param, next_all_action_parameters
            action = next_action
            state = next_state

            episode_reward += reward
            if visualise and i % render_freq == 0:
                env.render()

            if terminal:
                break
        agent.end_episode()

       # if save_frames and i % render_freq == 0:
       #     video_index = env.unwrapped.save_render_states(vidir, title, video_index)

        returns.append(episode_reward)
        total_reward += episode_reward
        if i % 100 == 0:
            print('{0:5s} R:{1:.4f} r100:{2:.4f}'.format(str(i), total_reward / (i + 1), np.array(returns[-100:]).mean()))
    end_time = time.time()
    print("Took %.2f seconds" % (end_time - start_time))
    env.close()
    if save_freq > 0 and save_dir:
        agent.save_models(os.path.join(save_dir, str(i)))

    returns = env.get_episode_rewards()
    print("Ave. return =", sum(returns) / len(returns))
    print("Ave. last 100 episode return =", sum(returns[-100:]) / 100.)

    np.save(os.path.join(dir, title + "{}".format(str(seed))),returns)

    if evaluation_episodes > 0:
        print("Evaluating agent over {} episodes".format(evaluation_episodes))
        agent.epsilon_final = 0.
        agent.epsilon = 0.
        agent.noise = None
        evaluation_returns = evaluate(env, agent, evaluation_episodes)
        print("Ave. evaluation return =", sum(evaluation_returns) / len(evaluation_returns))
        np.save(os.path.join(dir, title + "{}e".format(str(seed))), evaluation_returns)
        
    return sum(returns[-100:]) / 100.

def optimized_para(trail):
	return {
		'seed' : int(trail.suggest_int('seed', 1,2)), 
		'episodes' :  10000, 
		'evaluation_episodes' : 1000, 
		'batch_size' : int(trail.suggest_categorical('batch_size', ['128','64'])),  #128, 
		'gamma' : 0.9,
		'inverting_gradients' : True, 
		'initial_memory_threshold' : 500,
		'replay_memory_size' : 10000, 
		'epsilon_steps' : 1000, 
		'tau_actor' : 0.1, 
		'tau_actor_param' : 0.001, 
		'use_ornstein_noise' : True, 
		'learning_rate_actor' : float(trail.suggest_categorical('learning_rate_actor', ['0.01','0.001'])), #0.001,
		'learning_rate_actor_param' : float(trail.suggest_categorical('learning_rate_actor_param', ['0.0001','0.001'])), #0.0001, 
		'epsilon_final' : 0.01, 
		'zero_index_gradients' : False, 
		'initialise_params' : True, 
		'scale_actions' : True,
		'clip_grad' : 10.0, 
		'split' : False, 
		'indexed' : False, 
		'layers' : [128,], 
		'multipass' : True, 
		'weighted' : False, 
		'average' : False, 
		'random_weighted' : False, 
		'render_freq' : 100, 
		'save_freq' : 0, 
		'save_dir' : "results/platform", 
		'save_frames' : False, 
		'visualise' : True, 
		'action_input_layer' : 0,
		'title'  : "PDDQN"
		}


def optimize_hyperparm(trail):
        model_params =optimized_para(trail)
        return run(**model_params)
        
study = optuna.create_study(direction='maximize')

study.optimize(optimize_hyperparm, n_trials=4, n_jobs=1)
