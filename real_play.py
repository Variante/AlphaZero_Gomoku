# -*- coding: utf-8 -*-
"""
human VS AI models
Input your move in the format: 2,3

@author: Junxiao Song
"""

from __future__ import print_function
import pickle
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net_numpy import PolicyValueNetNumpy
from RobotManager import RobotManager
# from policy_value_net import PolicyValueNet  # Theano and Lasagne
# from policy_value_net_pytorch import PolicyValueNet  # Pytorch
# from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
# from policy_value_net_keras import PolicyValueNet  # Keras


class Human(object):
    """
    human player
    """

    def __init__(self, env):
        self.player = None
        self.env = env
        self.states = {0: [], 1: [], 2: []}

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        allow_cheat = False
        while True:
            cmds = [
                'c: calibrate camera;',
                'r: collect all start a new episode',
                'f: flip',
                'a: allow cheat',
            ]
            key = input("Press enter when you are ready.\n" + '\n'.join(cmds) + '\n')
            print('processing..')
            if key.startswith('c'):
                self.env.calibrate()
                continue
            elif key.startswith('f'):
                self.env.flip()
                continue
            elif key.startswith('r'):
                print('Start over')
                return -1
            elif key.startswith('a'):
                allow_cheat = True
            states = self.env.parse_board()
            diff = set(states[1]) - set(self.states[1])
            if len(diff) != 1 and not allow_cheat:
                print(states)
                print(self.states)
                print(f'Found {len(diff)} stones {diff}, please check again.')
                continue
            else:
                self.states = states
                return diff.pop()

    def __str__(self):
        return "Human {}".format(self.player)


def run():
    n = 4 # num of checkers to win
    width, height = 6, 6 # w and h of the checkerboard
    model_file = f'best_policy_{width}_{height}_{n}.model'
    human_first = True
    n_playout = 400 # difficulties
    
    with RobotManager(b_width=width, b_height=height) as env:
        # env.calibrate()
        env.capture_img()
        env.load_calib()
        # ############### human VS AI ###################
        # load the trained policy_value_net in either Theano/Lasagne, PyTorch or TensorFlow

        # best_policy = PolicyValueNet(width, height, model_file = model_file)
        # mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)

        # load the provided model (trained in Theano/Lasagne) into a MCTS player written in pure numpy
        try:
            policy_param = pickle.load(open(model_file, 'rb'))
        except:
            policy_param = pickle.load(open(model_file, 'rb'),
                                       encoding='bytes')  # To support python3
        best_policy = PolicyValueNetNumpy(width, height, policy_param)
        mcts_player = MCTSPlayer(best_policy.policy_value_fn,
                                 c_puct=5,
                                 n_playout=n_playout)  # set larger n_playout for better performance

        while True:
            try:
                board = Board(width=width, height=height, n_in_row=n)
                game = Game(board)
                # uncomment the following line to play with pure MCTS (it's much weaker even with a larger n_playout)
                # mcts_player = MCTS_Pure(c_puct=5, n_playout=1000)

                # human player, input your move in the format: 2,3
                human = Human(env)

                # set start_player=0 for human first
                game.start_play(human, mcts_player, env, start_player=0 if human_first else 0, is_shown=1)
                env.reset()
            except KeyboardInterrupt:
                print('\n\rquit')
                break


if __name__ == '__main__':
    run()
