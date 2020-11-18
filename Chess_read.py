'''
Authors: 	Yongyang Liu <liuyongyang@gatech.edu>
			Xiyang Wu <xwu391@gatech.edu>
			
Date: 		18 Nov 2020
'''
import numpy as np
import chess

# Read chess information
class CHESS_READ():
    def __init__(self):
        #states
        # 0 - empty; 1 - King; 2 - Queen; 3 - Bishop; 4 - Knight; 5 - Rook; 6 - Pawn;  7 - king; 8 - queen; 9 - bishop; 10 - knight; 11 - rook; 12 - pawn;
        self.piece        = ['.', 'K', 'Q', 'B', 'N', 'R', 'P', 'k', 'q', 'b', 'n', 'r', 'p']
        
        #actions
        square_x          = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        square_y          = ['1', '2', '3', '4', '5', '6', '7', '8']
        promotion         = ['', 'q']    # ['', 'q', 'r', 'b', 'n']   only consider promotion q
        self.action_space = [i+j+m+n+k for k in promotion for i in square_x for j in square_y for m in square_x for n in square_y if i+j != m+n]

        
    def read_state(self, board):
        # convert board information from Chess into 8x8 numpy  
        state_tmp = [self.piece.index(item) for item in str(board) if item != ' ' and item !='\n']
        state_tmp = np.array(state_tmp).reshape(8,8)   
        state = np.zeros((13,8,8))
        for i in range(8):
            for j in range(8):
                state[state_tmp[i,j],i,j] = 1.0         # assign probabiliry 1.0 to each piece in square
        return state
    
    def read_move(self, idx):
        return chess.Move.from_uci(self.action_space[idx])
    
    def read_idx(self, move): 
        if len(str(move)) > 4:
            move = ''.join([char for char in str(move)][0:4]) + 'q' # only consider promotion q
        else:
            move = str(move)
        return self.action_space.index(move)