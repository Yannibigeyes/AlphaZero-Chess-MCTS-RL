'''
Authors: 	Yongyang Liu <liuyongyang@gatech.edu>
			Xiyang Wu <xwu391@gatech.edu>
			
Date: 		18 Nov 2020
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CNN(nn.Module):
    def __init__(self, in_channels = 13, num_actions = 8064):  #4032*2
        super(CNN, self).__init__()
           
        self.in_channels   = in_channels
        self.num_actions   = num_actions
        self.out_channels1 = 64
        self.out_channels2 = 128
        self.out_channels3 = 256

        self.conv1   = nn.Conv2d(in_channels, self.out_channels1, 3)
        self.bn1     = nn.BatchNorm2d(self.out_channels1)
        self.conv2   = nn.Conv2d(self.out_channels1, self.out_channels2, 3)
        self.bn2     = nn.BatchNorm2d(self.out_channels2)
        self.conv3   = nn.Conv2d(self.out_channels2, self.out_channels3, 3)
        self.bn3     = nn.BatchNorm2d(self.out_channels3)
        
        self.wide    = self.feature_size(self.feature_size(self.feature_size(8, 3), 3), 3)
        #wide = 8 - 2 - 2 - 2 = 2
        
        self.in_fc1  = self.out_channels3 * self.wide * self.wide   #256 * 4 = 1024
        self.in_fc2  = 4096
        self.in_fc3  = 4096
        
        # policy layer: propability over action space
        self.fc1_pol = nn.Linear(self.in_fc1, self.in_fc2)
        self.fc2_pol = nn.Linear(self.in_fc2, self.in_fc3) 
        self.fc3_pol = nn.Linear(self.in_fc3, num_actions)
        
        # value layer: value for current state
        self.fc1_val = nn.Linear(self.in_fc1, 128)
        self.fc2_val = nn.Linear(128, 16) 
        self.fc3_val = nn.Linear(16, 1) 

        
    def forward(self, x):
        output = F.relu(self.bn1(self.conv1(x)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = F.relu(self.bn3(self.conv3(output)))
        output = output.view(-1, self.in_fc1)
        
        # policy layer: propability over action space 
        output_policy = F.relu(self.fc1_pol(output))
        output_policy = F.relu(self.fc2_pol(output_policy))
        output_policy = F.softmax(self.fc3_pol(output_policy), dim = 1)
        
        # value layer: value for current state
        output_value = F.relu(self.fc1_val(output))
        output_value = F.relu(self.fc2_val(output_value))
        output_value = self.fc3_val(output_value)

        return output_policy, output_value
   
    def feature_size(self, size, kernel_size, stride = 1, padding = 0):
        return int((size - kernel_size + 2*padding)/ stride  + 1)


class NEURAL_NETWORK():
    def __init__(self, learning_rate, PATH, device):
        self.learning_rate = learning_rate
        self.device        = device
        self.path          = PATH

        self.policy_net    = CNN().to(device)
        self.optimizer     = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        self.best_loss     = 99999999999     # initial threshold to save model
        self.L2_lambda     = 0.00001

    def optimize_model(self, buffer): 
        torch.cuda.empty_cache()
        
        # initialization
        state, probability, winner = buffer
        state_opt        = Variable(torch.Tensor(np.array(state))).to(self.device)
        probability_opt  = Variable(torch.Tensor(np.array(probability))).to(self.device)
        winner_opt       = Variable(torch.Tensor(np.array(winner))).to(self.device)
        
        # calculate by policy network
        act_probs, value = self.policy_net(state_opt)
        
        # minimizes the square error between the actual outcome of a game played, and the prediction of the winner from the neural network at each time step.
        value_loss       = F.mse_loss(value.view(-1), winner_opt)
        
        # make the following two probability distributions match, 
        # The probability distribution output by the neural network for the state s at time step t; 
        # The probability distribution determined by the MCTS for the probability of taking each action at time t in state s.
        policy_loss      = - torch.mean(torch.sum(probability_opt*(torch.log(act_probs)), 1))

        loss             = value_loss + policy_loss
        self.optimizer.zero_grad()

        # L2 regularization
        L2_reg           = torch.tensor(0., requires_grad=True)
        for name, param in self.policy_net.named_parameters():
            if 'weight' in name:
                L2_reg = L2_reg + torch.norm(param, 2)
        loss += self.L2_lambda * L2_reg
        
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.save_model(loss)
    
        return loss
    
    def save_model(self, loss): 
        if loss.item() < self.best_loss:
            torch.save({
                        'model_state_dict': self.policy_net.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': loss,
                        }, self.path)
            self.best_loss = loss.item()

    def load_model(self): 
        if self.device == "cpu":
            checkpoint = torch.load(self.path, map_location = torch.device('cpu'))
        else:
            checkpoint = torch.load(self.path)
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])