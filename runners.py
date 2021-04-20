import numpy as np
import tensorflow as tf

from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

import wandb

def run_dqn(config, config_name):
    from algorithms import DeepQLearning
    from players import SimpleRLPlayer
    from loggers import WandbLogger

    wandb.init(project="showdown-rl", tags=["tf2", "keras", "dqn", config_name])

    env_player = SimpleRLPlayer(battle_format="gen8randombattle")
    (opponent, second_opponent) = config

    model = Sequential([
    	Dense(128, activation="elu", input_shape=(1, 42)),
    	Flatten(),
    	Dense(64, activation="elu"),
    	Dense(len(env_player.action_space), activation="linear")
    ])

    policy = LinearAnnealedPolicy(
    	EpsGreedyQPolicy(),
    	attr="eps",
    	value_max=1.0,
    	value_min=0.05,
    	value_test=0,
    	nb_steps=10000,
    )

    memory = SequentialMemory(limit=10000, window_length=1)
    opt, metrics = Adam(lr=0.00025), ["mae"]
    gamma = 0.5

    exp = DeepQLearning(env_player, opponent, second_opponent, model, policy, memory, opt, metrics, gamma, WandbLogger)
    exp.train(20000)
    exp.eval(100)
