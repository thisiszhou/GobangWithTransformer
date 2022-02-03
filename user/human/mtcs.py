import numpy as np
from typing import Optional, List, Tuple
from chess_board.base_board import ChessBoard, GAME_PLAYER
from ai.net import Net


class TreeNode(object):
    def __init__(self, parent, p):
        self.parent = parent
        self.children = dict()
        self.n_visits = 0
        self._Q = 0
        self._P = p

    def expand(self, actions: List[Tuple[int, int]], probs: List[float]):
        assert len(actions) == len(probs)
        length = len(actions)
        for i in range(length):
            co, p = actions[i], probs[i]
            if co not in self.children.keys():
                self.children[co] = TreeNode(self, p)

    def select(self, gamma):
        return max(self.children.items(), key=lambda node: node[1].get_value(gamma))

    def get_value(self, gamma):
        u = gamma * self._P * np.sqrt(self.parent.n_visits) / (1 + self.n_visits)
        return self._Q + u

    def _update(self, value):
        self.n_visits += 1
        self._Q += 1.0 * (value - self.Q) / self.n_visits

    def update_recursive(self, value):
        if self.parent:
            self.parent.update_recursive(-value)
        self._update(value)

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None


class MCTS(object):
    def __init__(self, policy_net: Net, gamma=5, play_step=1000):
        self.root = TreeNode(None, 1)
        self.policy_net = policy_net
        self.gamma = gamma
        self.play_step = play_step

    def simulate_play(self, board: ChessBoard, current_player: GAME_PLAYER):
        node = self.root
        while not node.is_leaf():
            action, node = node.select(self.gamma)
            board.move(*action)


