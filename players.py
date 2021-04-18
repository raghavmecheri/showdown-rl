"""Various player class utilized during training"""

from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.player.random_player import RandomPlayer

import numpy as np

def _vectorise_stat(x):
    vects = {
    "atk": 0,
    "def": 1,
    "spa": 2,
    "spd": 3,
    "spe": 4,
    "accuracy": 5
    }
    return vects[x]

class SimpleRLPlayer(Gen8EnvSinglePlayer):
    def embed_battle(self, battle):
        moves_base_power = -np.ones(4)
        moves_base_status = np.zeros((4, 6))
        moves_base_heal = np.zeros(4)
        moves_dmg_multiplier = np.ones(4)

        for i, move in enumerate(battle.available_moves):
            moves_base_heal[i] = move.heal - move.recoil
            if not move.boosts == None:
                for k, v in move.boosts.items():
                    moves_base_status[i][_vectorise_stat(k)] += v

            moves_base_power[i] = (
                move.base_power / 100
                )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                )

        # We count how many pokemons have not fainted in each team
        remaining_mon_team = (
            len([mon for mon in battle.team.values() if mon.fainted]) / 6
        )
        remaining_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Final vector with 10 components
        return np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                moves_base_heal,
                moves_base_status.flatten(order='C'),
                [remaining_mon_team, remaining_mon_opponent],
            ]
        )

    def compute_reward(self, battle) -> float:
        return self.reward_computing_helper(
            battle, fainted_value=3, hp_value=1.5, status_value=0.75, victory_value=30
        )


class MaxDamagePlayer(RandomPlayer):
    def choose_move(self, battle):
        # If the player can attack, it will
        if battle.available_moves:
            # Finds the best move among available ones
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)


class RandomizedMaxDamagePlayer(RandomPlayer):
    def choose_move(self, battle):
        epsilon = 0.45
        if battle.available_moves:
            # Find the best move, but randomised
            if np.random.uniform() < epsilon:
                return self.choose_random_move(battle)

            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)
        else:
            return self.choose_random_move(battle)


class MiniMaxDamagePlayer(RandomPlayer):
    def choose_move(self, battle):
        return super().choose_move(battle)


class RandomizedMiniMaxDamagePlayer(RandomPlayer):
    def choose_move(self, battle):
        epsilon = 0.45
        if np.random.uniform() < epsilon:
            return super().choose_move(battle)
        else:
            return super().choose_move(battle)

