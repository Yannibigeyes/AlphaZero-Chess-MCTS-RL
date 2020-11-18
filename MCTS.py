'''
Authors: 	Yongyang Liu <liuyongyang@gatech.edu>
			Xiyang Wu <xwu391@gatech.edu>
			
Date: 		18 Nov 2020
'''
import numpy as np
import time
import chess
import torch
from torch.autograd import Variable

class TreeNode():
    def __init__(self, board, parent, probability):
        self.board    = board
        self.parent   = parent
        self.action   = None
        self.child    = {}
        
        self.N        = 0
        self.W        = 0
        self.Q        = 0
        self.P        = probability
        
        self.end_flag = False
        self.extended = False

class MCTS():
    def __init__(self, max_iter, NN, Chess_read, device):
        self.max_iter    = max_iter
        self.device      = device
        self.max_explor  = 600
        
        self.Net         = NN.policy_net
        self.Chess_read  = Chess_read
        self.Cpuct       = 5
        self.temperature = 1
        
        self.node_list   = {}
        self.node_list[str(chess.Board()), None] = TreeNode(chess.Board(), None, 1.0)
        
    def selection(self, node):      
        # select max (Q + U) to visit. then in play, the node with the highest N means max (Q+U)
        Q_U = []
        for action_taken in node.action:
            node_tmp = node.child[action_taken]
            U_value    = self.Cpuct * node_tmp.P * np.sqrt(node_tmp.parent.N) / (1 + node_tmp.N)
            Q_U.append(U_value + node_tmp.Q)
        idx_selected = np.argmax(np.array(Q_U))
        return node.action[idx_selected]

    def extension(self, node, probability):
        node.extended = True
        node.action   = list(node.board.legal_moves)
        
        # generate childern nodes for all legal actions
        for i in range(len(node.action)):
            new_board = node.board.copy()         
            new_board.push(node.action[i])
            idx       = self.Chess_read.read_idx(node.action[i])
            
            new_node  = TreeNode(new_board, node, probability[idx])
            new_node.end_flag = new_board.is_game_over()
            
            node.child[node.action[i]] = new_node
            self.node_list[str(new_board), node.action[i]] = new_node
        return node
        
    def back_pop(self, node, value):
        curr_node    = node
        curr_node.N += 1
        curr_node.W += value
        curr_node.Q  = curr_node.W / curr_node.N
        
        # back propagation along parents
        while curr_node:
            curr_node = curr_node.parent
            if curr_node:
                curr_node.N += 1
                curr_node.W += value
                curr_node.Q  = curr_node.W / curr_node.N
                   
                if sum([curr_node.child[act].N  for act in curr_node.action]) != curr_node.N:
                    print('warning (MCTS): total visit time N does not match between parent and children')
                
    def simulation(self):
        for iter_num in range(self.max_iter):
            print("MCTS-Iter:", iter_num)
            curr_board  = chess.Board()
            prev_action = None
            curr_node   = self.node_list[str(curr_board), prev_action]   # None is previous action
            time.sleep(0.1)
            
            for i in range(self.max_explor):
                # obtrain from Neural Network
                state_tmp          = Variable(torch.Tensor(self.Chess_read.read_state(curr_board).reshape(1,13,8,8))).to(self.device)
                probability, value = self.Net(state_tmp)
                
                if curr_node.extended:
                    curr_action = self.selection(curr_node)
                else:
                    if self.device == "cpu":
                        curr_node = self.extension(curr_node, probability.data.numpy()[0])
                    else:
                        curr_node = self.extension(curr_node, probability.data.cpu().numpy()[0])
                    # randomly choose an action
                    action_list = curr_node.action
                    idx_tmp = np.random.randint(len(action_list))
                    curr_action = curr_node.action[idx_tmp] 
                    
                curr_board.push(curr_action)
                curr_node = curr_node.child[curr_action]
                
                #prev_action = curr_action
                
                if str(curr_board) != str(curr_node.board):
                    print('warning (MCTS): board information does not match')
                  
                if curr_node.end_flag:
                    break
            
            self.back_pop(curr_node, value.item())   
            
    def play(self, curr_board, prev_action):
        node = self.node_list[str(curr_board), prev_action]
        Pi   = np.zeros(8064)                             
        if node.extended and node.N != 0:   
            # in play, once met a new node which was not extended in simulation, we would randomly choose an aciton and
            # extend this node. Its N is still zero. Next time, if we meet this node again, extended == True, but all N
            # is zero, we still have to randomly choose. i.e. node.N == 0 and node.extended==True
            for act in node.action:
                node_tmp = node.child[act]         
                idx = self.Chess_read.read_idx(act)
                Pi[idx] += (node_tmp.N)**(1/self.temperature)     # +=: all promotion would add to promotion q
            Pi = np.divide(Pi, Pi.sum())
            idx_act = np.argmax(Pi)
            if round(sum(Pi),1) != 1.0:
                print('warning (MCTS extended): Pi.sum()!= 1')                               
            return self.Chess_read.read_move(idx_act), Pi                             
        else:     
            action_list = list(curr_board.legal_moves)
            for i in range(len(action_list)):
                action = action_list[i]                 # not string, but 'move' object from Chess
                idx = self.Chess_read.read_idx(action)
                Pi[idx] += 1/len(action_list)
            idx_act = np.random.randint(len(action_list))
            if round(sum(Pi),1) != 1.0:
                print('warning (MCTS non_extended): Pi.sum()!= 1')
            node = self.extension(node, Pi)
            return action_list[idx_act], Pi               