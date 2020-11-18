'''
Authors: 	Yongyang Liu <liuyongyang@gatech.edu>
			Xiyang Wu <xwu391@gatech.edu>
			
Date: 		18 Nov 2020
'''
import chess
import numpy as np
import torch

from Chess_read import CHESS_READ
from Neural_Network import NEURAL_NETWORK
from MCTS import MCTS


class ALPHA():
    def __init__(self):
        #self.device = 'cpu' 
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.path = "best_policy.model"
        
        self.num_epoches     = 1
        self.train_iteration = 1
        self.mcts_iteration  = 1
        self.learning_rate   = 0.0001
        
        self.Chess_read = CHESS_READ()
        self.NN         = NEURAL_NETWORK(self.learning_rate, self.path, self.device)
        self.mcts_tree  = MCTS(self.mcts_iteration, self.NN, self.Chess_read, self.device)   # iteration length; Neural network; Chess read
                
        self.engine      = chess.engine.SimpleEngine.popen_uci(r"C:\Users\yliu3119\Desktop\AlphaZero\stockfish.exe")
        self.stockfish_flag = True
        
    def training(self): 
        for epoch in range(self.num_epoches):
            # self simulation by MCTS with current Neural Network
            self.mcts_tree.simulation()
            board      = chess.Board()
            move_count = 0
            move       = None
            states_current, propability = [], []
            
            # Play one game
            while not board.is_game_over():
                node_tmp = self.mcts_tree.node_list[str(board), move]
                if str(node_tmp.board) != str(board):
                    print('warning: node does not match board')
                    break
                states_current.append(self.Chess_read.read_state(board).reshape(13,8,8).copy())

                move, prob = self.mcts_tree.play(board, move)   # False - no stockfish; True - run stockfish
                if self.stockfish_flag:
                    result = self.engine.play(board, chess.engine.Limit(time=0.05))
                    move = result.move
                
                if move not in list(board.legal_moves):
                    print('warning: move is not legal')
                    break
                if round(sum(prob),1) != 1.0:
                    print('warning: Pi.sum()!= 1')
                    break
                propability.append(prob.copy())

                board.push(move)
                move_count += 1

            f = lambda x: 1 if x == '1-0' else (0 if x == '0-1' else 0.5)
            winner = [f(str(board.result())) for i in range(move_count)] 
            # 0 - winner is False; 1 - winner is True; 0.5 - is Draw
            
            # update neural Network
            buffer = (states_current, propability, winner)
            for itr in range(self.train_iteration):
                loss = self.NN.optimize_model(buffer)
                print("Epoch: %d/%d, Iter: %d/%d, loss: %1.12f" % (epoch + 1, self.num_epoches, itr + 1, self.train_iteration, loss.item()))

class VALIDATION():
    def __init__(self):
        #self.device = 'cpu' 
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.path = "best_policy.model"
        
        self.mcts_iteration = 100
        self.learning_rate  = 0.001
        
        self.Chess_read     = CHESS_READ()
        self.NN             = NEURAL_NETWORK(self.learning_rate, self.path, self.device)
        
        #Load saved model
        self.NN.load_model()
        self.mcts_tree      = MCTS(self.mcts_iteration, self.NN, self.Chess_read, self.device)   # iteration length; Neural network; Chess read

    def test(self): 
        self.mcts_tree.simulation()
        board      = chess.Board()
        move       = None
        while not board.is_game_over():
            node_tmp = self.mcts_tree.node_list[str(board), move]
            if str(node_tmp.board) != str(board):
                print('warning: node does not match board')
                break
                
            move, prob = self.mcts_tree.play(board, move)   # False - no stockfish; True - run stockfish
            if move not in list(board.legal_moves): 
                print(move)
                print('warning: move is not legal')
                break
            if round(sum(prob),1) != 1.0:
                print('warning: Pi.sum()!= 1')
                break
            board.push(move)

        return board

# training the model
model = ALPHA()
model.training()

# validate the pretrained model
validation = VALIDATION()
output = validation.test()

print(output)
print(output.result())