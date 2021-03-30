import numpy as np
from rl.agents.dqn import DQNAgent


class DeepQLearning:
    def __init__(self, env_player, opponent, second_opponent, model, policy, memory, opt, metrics, gamma):
        self.env_player = env_player
        self.opponent = opponent
        self.second_opponent = second_opponent

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

    def train(self, train_steps, save=False):
        self.env_player.play_against(
            env_algorithm=self._train,
            opponent=self.opponent,
            env_algorithm_kwargs={"dqn": self.dqn, "nb_steps": train_steps},
        )

        if save == True:
            model.save("model_%d" % train_steps)

    def eval(self, eval_eps):
        print("Results against train/first player:")
        self.env_player.play_against(
            env_algorithm=self._eval,
            opponent=self.opponent,
            env_algorithm_kwargs={"dqn": self.dqn, "nb_episodes": eval_eps},
        )

        print("\nResults against eval/second player:")
        self.env_player.play_against(
            env_algorithm=self._eval,
            opponent=self.second_opponent,
            env_algorithm_kwargs={"dqn": self.dqn, "nb_episodes": eval_eps},
        )

    def _train(self, player, dqn, nb_steps):
        dqn.fit(player, nb_steps=nb_steps)
        player.complete_current_battle()

    def _eval(self, player, dqn, nb_episodes):
        # Reset battle statistics
        player.reset_battles()
        dqn.test(player, nb_episodes=nb_episodes,
                 visualize=False, verbose=False)

        print("DQN Evaluation: {} victories out of {} episodes".format(
            player.n_won_battles, nb_episodes))
