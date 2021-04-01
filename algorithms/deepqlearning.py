import numpy as np
from rl.agents.dqn import DQNAgent
from rl.callbacks import WandbLogger
import wandb

class DeepQLearning:
	def __init__(self, env_player, opponent, second_opponent, model, policy, memory, opt, metrics, gamma):
		self.env_player = env_player
		self.opponent = opponent
		self.second_opponent = second_opponent

		wandb.init(project="showdown-rl", tags=["tf2", "keras", "dqn"])

		self.dqn = DQNAgent(
			model=model,
			nb_actions=len(env_player.action_space),
			policy=policy,
			memory=memory,
			nb_steps_warmup=1000,
			gamma=gamma,
			target_model_update=1,
			delta_clip=0.01,
			enable_double_dqn=True,
			)

		self.dqn.compile(opt, metrics)

	def train(self, train_steps, save = False):
		env_player.play_against(
			env_algorithm=self._train,
			opponent=self.opponent,
			env_algorithm_kwargs={"dqn": self.dqn, "nb_steps": train_steps},
			)

		if save == True:
			model.save("model_%d" % train_steps)

	def eval(self, eval_eps)
		print("Results against train/first player:")
		env_player.play_against(
			env_algorithm=self._eval,
			opponent=self.opponent,
			env_algorithm_kwargs={"dqn": self.dqn, "nb_episodes": eval_eps, "callbacks": [WandbLogger()]},
			)

		print("\nResults against eval/second player:")
		env_player.play_against(
			env_algorithm=self._eval,
			opponent=self.second_opponent,
			env_algorithm_kwargs={"dqn": self.dqn, "nb_episodes": eval_eps, "callbacks": [WandbLogger()]},
			)

	def _train(self, player, dqn, nb_steps, callbacks):
	    dqn.fit(player, nb_steps=nb_steps, callbacks=cbs)
	    player.complete_current_battle()


	def _eval(self, player, dqn, nb_episodes, callbacks=cbs):
	    # Reset battle statistics
	    player.reset_battles()
	    dqn.test(player, nb_episodes=nb_episodes, callbacks=cbs, visualize=False, verbose=False)

	    print("DQN Evaluation: %d victories out of %d episodes".format(player.n_won_battles, nb_episodes))
