# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from poke_env.player.random_player import RandomPlayer
from players import SimpleRLPlayer, MaxDamagePlayer

from runners import run_dqn

import argparse

tf.random.set_seed(42)
np.random.seed(42)

TRAIN_CONFIGS = {
	"rand_max": (RandomPlayer(battle_format="gen8randombattle"),
		MaxDamagePlayer(battle_format="gen8randombattle"))
}

MODEL_TRAINER_CONFIGS = {
	"dqn": run_dqn
}

def fetch_config(x):
	if x in TRAIN_CONFIGS:
		return TRAIN_CONFIGS[x]

	raise Exception("Passed config {} not defined".format(x))

def fetch_model_trainer(x):
	if x in MODEL_TRAINER_CONFIGS:
		return MODEL_TRAINER_CONFIGS[x]

	raise Exception("Passed model trainer config {} not defined".format(x))

parser = argparse.ArgumentParser(description='Training spec')
parser.add_argument("-m", "--model", required=True,
	help="model to train the RL agent using. Options: [\"dqn\"]")
parser.add_argument("-c", "--config", required=True,
	help="RL train/eval config. Options: [\"rand_max\"]")

if __name__ == "__main__":
	args = parser.parse_args()
	model, config = args["model"], args["config"]
	f, c = fetch_model_trainer(model), fetch_config(config)
	f(c)