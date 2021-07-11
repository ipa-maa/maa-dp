import os
import time
import gym
import gym_platform
from gym.wrappers import Monitor
from common.platform_domain import PlatformFlattenedActionWrapper
from common.wrappers import ScaledStateWrapper
from common.wrappers import ScaledParameterisedActionWrapper
import numpy as np
import optuna
import joblib


class HyperSearch:
    import optuna
    import yaml
    import logging
    import sys

    def __init__(self, model_optimization_function, Hyperparameter_path, search_size):
        """
        model_optimization_function ist die Funktion in der das Modell erzeugt und optimiert wird.
        Als erstes Argument bekommt diese Funktion den Optuna trial übergeben und anschließend alle Parameter

        Hyperparameter_path ist der Pfad der YML mit den Parametern

        search_size ist "big" pder "small" und beschreibt ob ein großes Parameterraum oder ein kleiner durchsucht wird
        """
        self.model_optimization_function = model_optimization_function
        self.Hyperparameter_path = Hyperparameter_path
        self.search_size = search_size
        self.save_dir = 'results/platform/'

    def optimized_para(self, trial):
        """
        Die Parameter werden aus der YML eingelesen und für Optuna die Trial auswahlen festgelegt.
        Return sind alle zu testenden Parameter
        """
        import yaml

        with open(self.Hyperparameter_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        n_layers = int(trial.suggest_categorical('n_layers', config[self.search_size]["n_layers"]))
        layers = []
        for i in range(n_layers):
            layers.append(int(trial.suggest_categorical('n_units_l{}'.format(i), config[self.search_size]["n_units_l"])))

        TrialID = str(trial.number)

        params = {
            'seed': 1,
            'episodes': 5000,
            'evaluation_episodes': 500,
            'batch_size': 128,
            'gamma': 0.9,
            'inverting_gradients': True,
            'initial_memory_threshold': 500,
            'replay_memory_size': 10000,
            'epsilon_steps': 1000,
            'tau_actor': 0.1,
            'tau_actor_param': 0.001,
            'use_ornstein_noise': True,
            'learning_rate_actor': 0.001,
            'learning_rate_actor_param': 0.0001,
            'epsilon_final': 0.01,
            'zero_index_gradients': False,
            'initialise_params': True,
            'scale_actions': True,
            'clip_grad': 10.0,
            'split': False,
            'indexed': False,
            'layers': [128, ],
            'weighted': False,
            'average': False,
            'random_weighted': False,
            'save_freq': 0,
            'save_dir': 'results/platform/',
            'save_frames': False,
            'action_input_layer': 0,
            'title': 'MPDDQN'}

        for i in params:
            if i == 'layers':
                params[i] = tuple(layers)
            elif i == 'save_dir':
                self.save_dir = config[self.search_size][i]
                params[i] = config[self.search_size][i]+TrialID
            elif type(config[self.search_size][i]) == list:
                params[i] = type(config[self.search_size][i][1])(trial.suggest_categorical(i, config[self.search_size][i]))
            else:
                params[i] = config[self.search_size][i]

        self.params = params

        return params

    def optimize_hyperparm(self, trial):
        """
        Die Parameter werden aufgerufen und der model_optimization_function übergeben
        """
        model_params = self.optimized_para(trial)

        if trial.should_prune():
            raise optuna.TrialPruned()

        return self.model_optimization_function(trial, **model_params)

    def optimize_study(self):
        """
        Der Sampler, zur auswahl der Parameter wird auf TPE Sampler gesetzt

        Der Puner, zum Abbruch von schlecht laufenden Trials wird auf Median Pruner mit
        5 Startup Runden und 10 warmup Steps gesetzt

        Der Returnwert der model_optimization_function soll über die Trials optimiert werden
        """

        # optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        self.study_name = "platform"+self.search_size  # Unique identifier of the study.
        self.storage_name = "sqlite:///{}.db".format(self.study_name)

        self.sampler = optuna.samplers.TPESampler(seed=10)

        self.n_startup_trials = 5
        self.n_warmup_steps = 10
        self.pruner = optuna.pruners.MedianPruner(n_startup_trials=self.n_startup_trials,
                                                  n_warmup_steps=self.n_warmup_steps)

        self.direction = 'maximize'

        if self.search_size == "big":
            self.n_trials = 120
        else:
            self.n_trials = 12

        self.study = optuna.create_study(sampler=self.sampler,
                                         direction=self.direction,
                                         pruner=self.pruner,
                                         study_name=self.study_name,
                                         storage=self.storage_name,
                                         load_if_exists=True
                                         )

        self.study.optimize(self.optimize_hyperparm, n_trials=self.n_trials, n_jobs=6)

        return self.study

    def study_plots(self, study):
        """
        Erzeugung und Speichern der Plots
        """

        from datetime import datetime
        from optuna.visualization import plot_intermediate_values
        from optuna.visualization import plot_parallel_coordinate
        from optuna.visualization import plot_param_importances
        from optuna.visualization import plot_slice

        timestr = datetime.now().strftime("%d%m%Y-%H%M%S")

        fig1 = plot_intermediate_values(study)
        fig2 = plot_param_importances(study)
        fig3 = plot_parallel_coordinate(study)
        fig4 = plot_slice(study)

        save_dir_fig1 = self.save_dir + timestr + "-intermediate_values.html"
        save_dir_fig2 = self.save_dir + timestr + "-param_importances.html"
        save_dir_fig3 = self.save_dir + timestr + "-parallel_coordinate.html"
        save_dir_fig4 = self.save_dir + timestr + "-slice.html"

        fig1.write_html(save_dir_fig1)
        fig2.write_html(save_dir_fig2)
        fig3.write_html(save_dir_fig3)
        fig4.write_html(save_dir_fig4)

        fig1.show()
        fig2.show()
        fig3.show()
        fig4.show()


def pad_action(act, act_param):
    params = [np.zeros((1,), dtype=np.float32), np.zeros((1,),
                                                         dtype=np.float32), np.zeros((1,), dtype=np.float32)]

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


def run(trial, seed, episodes, evaluation_episodes, batch_size, gamma,
        inverting_gradients, initial_memory_threshold, replay_memory_size,
        epsilon_steps, tau_actor, tau_actor_param, use_ornstein_noise,
        learning_rate_actor, learning_rate_actor_param, epsilon_final,
        zero_index_gradients, initialise_params, scale_actions,
        clip_grad, split, indexed, layers, weighted, average, random_weighted,
        save_freq, save_dir, save_frames, action_input_layer, title):

    TrialID = str(trial.number)

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

    dir = os.path.join(save_dir, title)
    env = Monitor(env, directory=os.path.join(dir, str(seed)),
                  video_callable=False, write_upon_reset=False, force=True)
    env.seed(seed)
    np.random.seed(seed)

    print("Trail ID: " + TrialID, env.observation_space)

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
                      'action_input_layer': action_input_layer, },
        actor_param_kwargs={'hidden_layers': layers,
                            'squashing_function': False,
                            'output_layer_init_std': 0.0001, },
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
    ave_evaluations = []
    # video_index = 0
    # agent.epsilon_final = 0.
    # agent.epsilon = 0.
    # agent.noise = None

    for i in range(episodes):
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32, copy=False)

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

            if terminal:
                break
        agent.end_episode()

        returns.append(episode_reward)
        total_reward += episode_reward
        if i % 100 == 0:
            print('{0:5s} R:{1:.4f} r100:{2:.4f}'.format(str(i),
                                                         total_reward / (i + 1),
                                                         np.array(returns[-100:]).mean()))

        epsilon = agent.epsilon
        noise = agent.noise

        if i % evaluation_episodes == 0:
            agent.epsilon_final = 0.
            agent.epsilon = 0.
            agent.noise = None

            evaluation_returns = evaluate(env, agent, evaluation_episodes)
            print("Trial ID: " + TrialID, "Ave. evaluation return =",
                  sum(evaluation_returns) / len(evaluation_returns))
            ave_evaluations.append(sum(evaluation_returns) /
                                   len(evaluation_returns))

            agent.epsilon_final = epsilon_final
            agent.epsilon = epsilon
            agent.noise = noise

            trial.report(ave_evaluations[-1], i)

    end_time = time.time()
    print("Took %.2f seconds" % (end_time - start_time))
    env.close()

    returns = env.get_episode_rewards()
    print("Ave. return =", sum(returns) / len(returns))
    print("Ave. last 100 episode return =", sum(returns[-100:]) / 100.)

    if evaluation_episodes > 0:
        agent.epsilon_final = 0.
        agent.epsilon = 0.
        agent.noise = None
        evaluation_returns = evaluate(env, agent, evaluation_episodes)
        print("Ave. evaluation return =",
              sum(evaluation_returns) / len(evaluation_returns))

    return sum(evaluation_returns) / len(evaluation_returns)


Hyperparameter_path = '/home/dpotyka/maa-dp/Parameters.yml'


hyper_test = HyperSearch(run, Hyperparameter_path, "small")

small_study = hyper_test.optimize_study()

joblib.dump(small_study, "small_study.pkl")

# small_study = joblib.load("small_study.pkl")


hyper_test.study_plots(small_study)
