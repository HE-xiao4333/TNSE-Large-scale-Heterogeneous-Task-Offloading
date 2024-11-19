import random
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from config import config
import torch.nn.init as init
N =config.get('Dev_dev')
curPath = config.get("curPath")
Datasets_path = config.get("Datasets_path")
epoch_all = config.get("epoch")
Model_path = config.get("Model_path")
Result_path = config.get("Result_path")

class LSTM_Cell(nn.Module):
    def __init__(self, in_dim , hidden_dim):
        super(LSTM_Cell,self).__init__()
        self.ix_linear = nn.Linear(in_dim,hidden_dim)
        self.ih_linear = nn.Linear(hidden_dim,hidden_dim)
        self.fx_linear = nn.Linear(in_dim, hidden_dim)
        self.fh_linear = nn.Linear(hidden_dim, hidden_dim)
        self.ox_linear = nn.Linear(in_dim, hidden_dim)
        self.oh_linear = nn.Linear(hidden_dim, hidden_dim)
        self.cx_linear = nn.Linear(in_dim, hidden_dim)
        self.ch_linear = nn.Linear(hidden_dim, hidden_dim)
    def forward(self,x , h_1,c_1):
        i = torch.sigmoid(self.ix_linear(x)+self.ih_linear(h_1))
        f = torch.sigmoid(self.fx_linear(x)+self.fh_linear(h_1))
        o = torch.sigmoid(self.ox_linear(x) + self.oh_linear(h_1))
        c_ = torch.tanh(self.cx_linear(x) + self.ch_linear(h_1))
        c = i * c_ + f *c_1
        h = o * torch.tanh(c) 
        h = torch.sigmoid(h)
        return  h , c

class LSTM(nn.Module):
    def __init__(self, in_dim , hidden_dim):
        super(LSTM,self).__init__()
        self.hidden_dim =hidden_dim
        self.lstm_cell = LSTM_Cell(in_dim , hidden_dim)
    def forward(self,x):
        '''
        x = [seq_lens, batch_size, in_dim]
        '''
        outs=[]
        h,c =None,None
        for seq_x in x:
            #seq_x : [batch,in_dim]
            if h is None: h = torch.randn(1,self.hidden_dim)
            if c is None: c = torch.randn(1,self.hidden_dim)
            h,c = self.lstm_cell(seq_x,h,c)
            outs.append(torch.unsqueeze(h,0))
        outs = torch.cat(outs)
        return outs,h

class Actor(nn.Module):
    def __init__(self,state_dim,action_dim):
        super(Actor,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Sigmoid()
        )
        nn.init.normal_(self.model[0].weight, 0., 0.3)
        nn.init.constant_(self.model[0].bias, 0.1)
        if torch.cuda.is_available():
            self.model.cuda()
    def forward(self,h):
        if torch.cuda.is_available():
            h = h.to("cuda")
        m_pred = self.model(h)
        #m_pred = m_pred.detach().numpy()
        return m_pred
class Critic(nn.Module):
    def __init__(self,state_dim_RSU,state_dim_VE,learning_rate=0.001,training_interval=5,batch_size=32,memory_size=10000,):
        super(Critic,self).__init__()
        self.state_dim_RSU=state_dim_RSU
        self.state_dim_VE = state_dim_VE
        self.learning_rate=learning_rate
        self.training_interval = training_interval
        self.memory_size=memory_size
        self.batch_size=batch_size
        self.critic_RSU = self._build_net(self.state_dim_RSU)
        self.optimizer_RSU = optim.Adam(self.critic_RSU.parameters(), lr=self.learning_rate, betas=(0.09, 0.999),
                                   weight_decay=0.0001)
        self.memory_RSU = np.zeros((self.memory_size, state_dim_RSU + 1))
        self.memory_RSU_counter = 0
        self.cost_RSU = []
        self.Critic_VE=[]
        self.Optimizer_VE=[]
        self.Cost_VE=[]
        self.memory_VE = np.zeros((self.memory_size, state_dim_VE + 1))
        self.memory_VE_zero = np.zeros((self.memory_size, state_dim_VE + 1))
        for dev in range(config.get('Dev_dev')):
            critic_ve = self._build_net(self.state_dim_VE)
            optimizer_ve = optim.Adam(critic_ve.parameters(), lr=self.learning_rate, betas=(0.09, 0.999),
                                       weight_decay=0.0001)
            self.Critic_VE.append(critic_ve)
            self.Optimizer_VE.append(optimizer_ve)
            self.Cost_VE.append([])
        self.memory_VE_counter = 0
        self.memory_VE_counter_zero = 0
        self.cost=0


    def _build_net(self,h):
        model = nn.Sequential(
            nn.Linear(h, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Identity()
        )
        nn.init.normal_(model[0].weight, 0., 0.3)
        nn.init.constant_(model[0].bias, 0.1)
        if torch.cuda.is_available():
            model.cuda()
        return model

    def decode_RSU(self,h):
        h = torch.FloatTensor(h)
        if torch.cuda.is_available():
            h = h.to("cuda")
        self.model.eval()
        reward = self.critic_RSU(h)
        if torch.cuda.is_available():
            reward = reward.cpu()
        reward =reward.detach().numpy()
        return reward

    def decode_VE(self,dev,h):
        h = torch.FloatTensor(h)
        if torch.cuda.is_available():
            h = h.to("cuda")
        self.model.eval()
        reward = self.Critic_VE[dev](h)
        if torch.cuda.is_available():
            reward = reward.cpu()
        reward =reward.detach().numpy()
        return reward

    def remember_RSU(self,h,m,r):
        idx = self.memory_RSU_counter % self.memory_size
        self.memory_RSU[int(idx), :] = np.hstack((h, m, r))
        self.memory_RSU_counter += 1
        if self.memory_RSU_counter % self.training_interval == 0:
            self.learn_RSU()

    def learn_RSU(self):
        if self.memory_RSU_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_RSU_counter, size=self.batch_size)
        batch_memory = self.memory_RSU[sample_index, :]
        h_train = torch.Tensor(batch_memory[:, 0: self.state_dim_RSU])
        m_train = torch.Tensor(batch_memory[:, -1:])
        if torch.cuda.is_available():
            h_train = h_train.to("cuda")
            m_train = m_train.to("cuda")
        criterion = nn.MSELoss()
        self.critic_RSU.train()
        self.optimizer_RSU.zero_grad()
        predict = self.critic_RSU(h_train)
        loss = criterion(predict, m_train)
        loss.backward()
        self.optimizer_RSU.step()
        self.cost = loss.item()
        self.cost = loss.item()
        assert (self.cost >= 0)
        self.cost_RSU.append(self.cost)

    def remember_VE(self,h,m,r):#state+action
        if r == 0:
            idx = self.memory_VE_counter_zero % self.memory_size
            self.memory_VE_zero[int(idx), :] = np.hstack((h, m, r))
            self.memory_VE_counter_zero +=1
        else:
            idx = self.memory_VE_counter % self.memory_size
            self.memory_VE[int(idx), :] = np.hstack((h, m, r))
            self.memory_VE_counter += 1
            if self.memory_VE_counter % self.training_interval == 0:
                self.learn_VE()

    def learn_VE(self):
        for dev in range(config.get('Dev_dev')):
            if self.memory_VE_counter > self.memory_size:
                sample_index = np.random.choice(self.memory_size, size=self.batch_size)
            else:
                sample_index = np.random.choice(self.memory_VE_counter, size=self.batch_size)
            if self.memory_VE_counter_zero > self.memory_size:
                sample_index_zero = np.random.choice(self.memory_size, size=self.batch_size)
            else:
                sample_index_zero = np.random.choice(self.memory_VE_counter_zero, size=self.batch_size)
            batch_memory1 = self.memory_VE[sample_index, :]
            batch_memory2 = self.memory_VE_zero[sample_index_zero, :]
            batch_memory = np.concatenate((batch_memory1,batch_memory2),axis=0)
            h_train = torch.Tensor(batch_memory[:, 0: self.state_dim_VE])
            m_train = torch.Tensor(batch_memory[:, -1:])
            if torch.cuda.is_available():
                h_train = h_train.to("cuda")
                m_train = m_train.to("cuda")
            criterion = nn.MSELoss()
            self.Critic_VE[dev].train()
            self.Optimizer_VE[dev].zero_grad()
            predict = self.Critic_VE[dev](h_train)
            loss = criterion(predict, m_train)
            loss.backward()
            self.Optimizer_VE[dev].step()
            self.cost = loss.item()
            assert (self.cost >= 0)
            self.Cost_VE[dev].append(self.cost)


    def Loss_get(self):
        return self.cost_RSU,self.Cost_VE

class multiagent(object):
    def __init__(self,vehicle_number,RSU_number,state_dim_RSU,action_dim_RSU,state_dim_ve,action_dim_ve,LSTM_ve,LSTM_RSU,learning_rate=0.001, training_interval=5, batch_size=64,memory_size=10000):
        super(multiagent, self).__init__()
        self.vehicle = vehicle_number
        self.state_dim_ve = state_dim_ve#1+4+10+(2+2*3+1+3)+(1+1+2*3+3*3)+(1+1+4+2+3)
        self.action_dim_ve = action_dim_ve#2
        self.RSU = RSU_number
        self.state_dim_RSU = state_dim_RSU#1+(2+1+1)+(1+2+3+4)+MES*(1+1+3*2)+
        self.action_dim_RSU = action_dim_RSU#1+1
        self.LSTM_ve = LSTM_ve
        self.LSTM_RSU = LSTM_RSU #(1+1+3+4+MES)
        self.memory_size = memory_size
        self.training_interval = training_interval  # learn every #training_interval
        self.lr = learning_rate
        self.batch_size = batch_size
        # reply buffer
        self.memory_ve= np.zeros((self.memory_size, state_dim_ve + action_dim_ve + 1))
        self.point_ve = 0
        self.memory_RSU = np.zeros((self.memory_size, state_dim_RSU + action_dim_RSU))
        self.point_RSU = 0
        # 初始化LSTM
        self.LSTM_L = LSTM(LSTM_ve, LSTM_ve)  # lastest 5 second information --->input: output:5*1*(1+1+4+3+2)11
        self.optimizer_L = torch.optim.Adam(self.LSTM_L.parameters(), lr=0.001, betas=(0.09, 0.999),weight_decay=0.0001)
        #self.memory_LSTM_L = np.zeros((5, LSTM_ve))
        # 初始化Actor
        self.Actor_ve = []
        self.Actor_ve_target = []
        self.copt_1 = []
        self.copt_2 = []
        self.cost_ve = []
        for _ in range(self.vehicle):
            network = Actor(state_dim_ve, action_dim_ve)#primary_actor
            network_target = Actor(state_dim_ve, action_dim_ve)#target_actor
            copt = torch.optim.Adam(network.parameters(), lr=0.001, betas=(0.09, 0.999), weight_decay=0.0001)
            copt_target = torch.optim.Adam(network_target.parameters(), lr=0.001, betas=(0.09, 0.999),weight_decay=0.0001)
            self.Actor_ve.append(network)
            self.Actor_ve_target.append(network_target)
            self.copt_1.append(copt)
            self.copt_2.append(copt_target)
            self.cost_ve.append([])

        
        self.Actor_RSU = Actor(state_dim_RSU, action_dim_RSU)
        self.Actor_RSU_target = Actor(state_dim_RSU, action_dim_RSU)
        self.aopt = torch.optim.Adam(self.Actor_RSU.parameters(), lr=0.001, betas=(0.09, 0.999), weight_decay=0.0001)
        self.aopt_target = torch.optim.Adam(self.Actor_RSU_target.parameters(), lr=0.001, betas=(0.09, 0.999), weight_decay=0.0001)
        self.cost_RSU=[]

        
        self.mes_loss = nn.MSELoss()
        self.update_Actor = 0
    def predict_L(self,memory_LSTM_L):
        memory_LSTM_L_tensor = torch.tensor(memory_LSTM_L, dtype=torch.float32)
        self.LSTM_L.eval()
        _,r =self.LSTM_L(memory_LSTM_L_tensor)
        if torch.cuda.is_available():
            r = r.cpu()
        r = r.detach().numpy()
        return r
    def update_para_LSTM(self,input,real_L):
        input = torch.tensor(input, dtype=torch.float32)
        real_L = torch.tensor(real_L)
        criterion = nn.BCELoss()
        self.LSTM_L.train()
        self.optimizer_L.zero_grad()
        _,predict_L = self.LSTM_L(input)
        real_L = real_L.to(predict_L.dtype)
        real_L =real_L.view(predict_L.shape)
        loss = criterion(predict_L, real_L)
        loss.backward()
        self.optimizer_L.step()


    def remember_RSU(self,state,action):
        idx = self.point_RSU % self.memory_size
        self.memory_RSU[int(idx), :] = np.hstack((state, action))
        self.point_RSU +=1
        if self.point_RSU % (self.training_interval * N) == 0:
            self.learn_RSU()

    def remember_Vehile(self,ID,state,action,r):
        point_v=self.point_ve
        idx = self.point_ve % self.memory_size
        if r > 0:
            self.memory_ve[int(idx), :] = np.hstack((ID, state, action))
            self.point_ve += 1
        if (self.point_ve % (self.training_interval*config.get("Dev_dev")) == 0) & (self.point_ve>0)  & (self.point_ve != point_v):
            self.learn_ve()

    def choose_action_RSU(self, s, time):
        s = torch.FloatTensor(s)
        self.Actor_RSU.eval()
        action_all=[]
        action = self.Actor_RSU(s)
        if torch.cuda.is_available():
            action = action.cpu()
        action = action.detach().numpy()
        action_all.append(action.tolist())
        action_target = self.reverse(action,time)
        return action_target

    def choose_action_Vehicle(self, j,s, time):
        s = torch.FloatTensor(s)
        self.Actor_ve[j].eval()
        action_all=[]
        action = self.Actor_ve[j](s)
        if torch.cuda.is_available():
            action = action.cpu()
        action = action.detach().numpy()
        action_all.append(action.tolist())
        action_target = self.reverse(action,time)
        return action_target

    def learn_RSU(self):
        if self.point_RSU > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.point_RSU, size=self.batch_size)
        batch_memory = self.memory_RSU[sample_index, :]
        h_train = torch.Tensor(batch_memory[:, 0: self.state_dim_RSU])
        m_train = torch.Tensor(batch_memory[:, self.state_dim_RSU:])
        if torch.cuda.is_available():
                h_train = h_train.to("cuda")
                m_train = m_train.to("cuda")
        criterion = nn.CrossEntropyLoss()
        self.Actor_RSU.train()
        self.aopt_target.zero_grad()
        predict = self.Actor_RSU(h_train)
        loss = criterion(predict, m_train)
        loss.backward()
        self.aopt_target.step()
        self.cost = loss.item()
        assert (self.cost > 0)
        self.cost_RSU.append(self.cost) #
        tau = 0.999
        for param, target_param in zip(self.Actor_RSU.parameters(), self.Actor_RSU_target.parameters()):
            target_param.data.copy_((1 - tau) * target_param.data + tau * param.data)
        for param, target_param in zip(self.Actor_RSU.parameters(), self.Actor_RSU_target.parameters()):
            param.data.copy_(target_param.data)
        if self.point_RSU % (5 * self.training_interval*config.get("Dev_dev")) == 0:
            nn.init.normal_(self.Actor_RSU.model[-2].weight, mean=0., std=0.3)
            nn.init.constant_(self.Actor_RSU.model[-2].bias, 0.1)
            self.aopt_target = torch.optim.Adam(self.Actor_RSU.parameters(), lr=0.001, betas=(0.09, 0.999),
                                                weight_decay=0.0001)

    def learn_ve(self):
        for i in range(config.get('Dev_dev')):
            if self.point_ve > self.memory_size:
                sample_index = np.random.choice(self.memory_size, size=self.batch_size)
            else:
                sample_index = np.random.choice(self.point_ve, size=self.batch_size)
            batch_memory = self.memory_ve[sample_index, :]
            h_train = torch.Tensor(batch_memory[:, 1: self.state_dim_ve+1])
            m_train = torch.Tensor(batch_memory[:, self.state_dim_ve+1:])
            if torch.cuda.is_available():
                h_train = h_train.to("cuda")
                m_train = m_train.to("cuda")
            criterion = nn.BCELoss()
            self.Actor_ve[i].train()
            self.copt_2[i].zero_grad()
            predict = self.Actor_ve[i](h_train)
            loss = criterion(predict, m_train)
            loss.backward()
            self.copt_2[i].step()
            self.cost = loss.item()
            assert (self.cost > 0)
            self.cost_ve[i].append(self.cost)  #
            tau = 0.999
            for param, target_param in zip(self.Actor_ve[i].parameters(), self.Actor_ve_target[i].parameters()):
                target_param.data.copy_((1 - tau) * target_param.data + tau * param.data)
            for param, target_param in zip(self.Actor_ve[i].parameters(), self.Actor_ve_target[i].parameters()):
                param.data.copy_(target_param.data)
            if self.point_ve % (5 * self.training_interval*config.get("Dev_dev")) == 0:
                nn.init.normal_(self.Actor_ve[i].model[-2].weight, mean=0., std=0.3)
                nn.init.constant_(self.Actor_ve[i].model[-2].bias, 0.1)
                self.copt_2[i] = torch.optim.Adam(self.Actor_ve[i].parameters(), lr=0.001, betas=(0.09, 0.999),
                                                  weight_decay=0.0001)
            

    def get_cost(self):
        return self.cost_RSU,self.cost_ve

    def reverse(self, m, time,k=2):
        m_list = []
        if k > 0:
            for i in range(k):
                c = []
                
                nu = np.random.uniform(-0.25, 0.25)
                for j in range(len(m)):
                    if (m[j] + nu < 0):
                        index = 0.0001
                    elif (m[j] + nu > 1):
                        index = 0.9999
                    else:
                        index = m[j] + nu
                    c.append(index)
                m_list.append(c)
            pass
        return m_list


class SimpleNeuralNetwork(nn.Module):
    def __init__(self,input,output):
        super(SimpleNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x
class unit_all(object):
    def __init__(self,state_dim_input_user,state_dim_output_user,state_dim_input_mes,state_dim_output_mes):
        super(unit_all, self).__init__()
        self.model_user = SimpleNeuralNetwork(state_dim_input_user,state_dim_output_user)
        self.model_mes = SimpleNeuralNetwork(state_dim_input_mes,state_dim_output_mes)
        #criterion = nn.BCELoss()
        optimizer_uer = optim.Adam(self.model_user.parameters(), lr=0.001, betas=(0.09, 0.999),weight_decay=0.0001)
    def get_output_user(self,input):
        input = torch.tensor(input, dtype=torch.float32)
        self.model_user.eval()
        output = self.model_user(input)
        if torch.cuda.is_available():
            output = output.cpu()
        output = output.detach().numpy()
        return output
    def get_output_mes(self,input):
        input = torch.tensor(input, dtype=torch.float32)
        self.model_mes.eval()
        output = self.model_mes(input)
        if torch.cuda.is_available():
            output = output.cpu()
        output = output.detach().numpy()
        return output
