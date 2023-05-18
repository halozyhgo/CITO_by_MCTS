import torch
from torch.distributions import MultivariateNormal
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import time
import math, random
import numpy as np
import sys
import os
import csv
import shutil
import operator
import datetime
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import deque
from DuelingDQN import DuelingDQN
import LoadSootdata
from NaivePrioritizedBuffer import NaivePrioritizedBuffer
from TClass import tclass
from TEdge import tedge
from TSGenv import env
from d2l import torch as d2l
#####################  hyper parameters  ####################

device = d2l.try_gpu()
MAX_EPISODES = 400               # 最大训练代数
MAX_EP_STEPS = 200               # episode最大持续帧数

RENDER = False
ENV_NAME = 'Pendulum-v0'         # 游戏名称
SEED = 123                       # 随机数种子


class DDPG(object):
    def __init__(self,a_dim,s_dim,):
        self.a_dim = a_dim
        self.s_dim = s_dim
        self.lr_actor = 0.001
        self.lr_critic = 0.001
        self.gamma = 0.9        # 折扣率

        # Initialize the covariance matrix used to query the actor for actions
        self.cov_var = torch.full(size=(self.a_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        class ANet(nn.Module):      # 动作网络
            def __init__(self,s_dim,a_dim):
                super(ANet, self).__init__()
                self.fc1 = nn.Linear(s_dim,64)
                self.fc1 = nn.Linear(s_dim, 64)
                self.dp = nn.Dropout(p=0.5)
                self.out = nn.Linear(64,a_dim)

            def forward(self,x):

                x = self.fc1(x)
                x = F.relu(x)
                x = self.dp(x)
                x = F.tanh(x)
                x = self.out(x)
                actions_value = x # * self.a_bound.item()

                return actions_value

        class CNet(nn.Module):  # 定义价值网络
            def __init__(self, s_dim):
                super(CNet, self).__init__()
                self.fcs = nn.Linear(s_dim, 64)
                self.fcs.weight.data.normal_(0, 0.1)  # initialization
                self.out = nn.Linear(64, 1)
                self.out.weight.data.normal_(0, 0.1)  # initialization

            def forward(self, s):
                x = self.fcs(s)  # 输入状态
                # y = self.fca(a)  # 输入动作
                # net = F.relu(x + y)
                actions_value = self.out(x)  # 给出V(s,a)
                return actions_value

        # 这个可以放在训练网络方法中，在开始训练之前对网络中的参数进行初始化
        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                nn.init.xavier_uniform_(m.weight)

        self.Actor_eval = ANet(s_dim, a_dim).to(device)  # 主网络
        self.Actor_target = ANet(s_dim, a_dim, ).to(device)  # 目标网络
        self.Critic_eval = CNet(s_dim).to(device)  # 主网络
        self.Critic_target = CNet(s_dim).to(device)  # 当前网络

        # 参数初始化
        self.Actor_eval.apply(init_weights)
        self.Actor_target.apply(init_weights)
        self.Critic_eval.apply(init_weights)
        self.Critic_target.apply(init_weights)

        self.critic_train = torch.optim.Adam(self.Critic_eval.parameters(), lr=self.lr_actor)  # critic的优化器
        self.actor_train = torch.optim.Adam(self.Actor_eval.parameters(), lr=self.lr_critic)  # actor的优化器
        self.loss_td = nn.MSELoss()  # 损失函数采用均方误差

    def evaluate(self,batch_obs, batch_acts,batch_transfer):
        """
            Estimate the values of each observation, and the log probs of
            each action in the most recent batch with the most recent
            iteration of the actor network. Should be called from learn.
            Parameters:
                batch_obs - the observations from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of observation)
                batch_acts - the actions from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of action)
            Return:
                V - the predicted values of batch_obs
                log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
        """
        # Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        V = self.Critic_eval(batch_obs).squeeze()

        # Calculate the log probabilities of batch actions using most recent actor network.
        # This segment of code is similar to that in get_action()
        mean = self.Actor_eval(batch_obs)
        mean = mean.mul(batch_transfer)
        dist = torch.distributions.Categorical(mean)
        log_probs = dist.log_prob(batch_acts)

        # Return the value vector V of each observation in the batch
        # and log probabilities log_probs of each action in the batch
        return V, log_probs

    def get_action(self,state):
        transfer = []
        for item in state:
            if item != -1:
                transfer.append(0)
                continue
            transfer.append(1)
        action_value = torch.FloatTensor(state).to(device)
        action_value = self.Actor_eval(action_value)
        action_value = F.softmax(action_value)
        transfer1 = torch.tensor(transfer).to(device)
        action_value = action_value.mul(transfer1)
        dist = torch.distributions.Categorical(action_value)
        action = dist.sample()
        # Calculate the log probability for that action
        log_prob = dist.log_prob(action)

        # Return the sampled action and the log probability of that action in our distribution
        return action.item(), log_prob.detach(),transfer

    def compute_rtgs(self,batch_rews):
        """
        Compute the Reward-To-Go of each timestep in a batch given the rewards.
        Parameters:
            batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)
        Return:
            batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
        """
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []

        # Iterate through each episode
        for ep_rews in reversed(batch_rews):

            discounted_reward = 0  # The discounted reward so far

            # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
            # discounted return (think about why it would be harder starting from the beginning)
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs

    def update_model(self,batch_obs,batch_acts,batch_log_probs,batch_rews,batch_trans):
        # Reshape data as tensors in the shape specified in function description, before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float).to(device)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float).to(device)
        batch_trans = torch.tensor(batch_trans, dtype=torch.float).to(device)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float).to(device)
        batch_rtgs = self.compute_rtgs(batch_rews).to(device)
        V, _ = self.evaluate(batch_obs, batch_acts,batch_trans)
        A_k = batch_rtgs - V.detach()

        A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

        for _ in range(5):  # ALG STEP 6 & 7
            # Calculate V_phi and pi_theta(a_t | s_t)
            V, curr_log_probs = self.evaluate(batch_obs, batch_acts,batch_trans)

            ratios = torch.exp(curr_log_probs - batch_log_probs)
            # Calculate surrogate losses.
            surr1 = ratios * A_k
            #                           1 - epsilon, 1+epsilon
            surr2 = torch.clamp(ratios, 1 - 0.2, 1 + 0.2) * A_k

            actor_loss = (-torch.min(surr1, surr2)).mean()
            critic_loss = nn.MSELoss()(V, batch_rtgs)
            # Calculate gradients and perform backward propagation for actor network
            self.actor_train.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_train.step()

            # Calculate gradients and perform backward propagation for critic network
            self.critic_train.zero_grad()
            critic_loss.backward()
            self.critic_train.step()

#将excel信息导入环境
def info2env(filename,mode=0):
    nclass=LoadSootdata.getNClass(filename)
    path=list()
    couplematrix=list()
    mapoftclass=dict()
    setofedge=list()
    if mode:#导入动态情况的信息
        path=LoadSootdata.getDynamicPath(filename,nclass)#获取动态的path信息
        couplematrix=LoadSootdata.getCoupleMatrix(filename,nclass,mode)#获取耦合矩阵
        mapoftclass=LoadSootdata.getMAPofTClass(filename)#TClass的映射
        setofedge=LoadSootdata.getSETofTEdge(filename,nclass,mapoftclass,mode)#获取动态情况的边集合
    else:#导入静态情况的信息
        path=LoadSootdata.getStaticPath(filename,nclass)#获取静态的path信息
        couplematrix=LoadSootdata.getCoupleMatrix(filename,nclass,mode)#获取耦合矩阵
        mapoftclass=LoadSootdata.getMAPofTClass(filename)#TClass的映射
        setofedge=LoadSootdata.getSETofTEdge(filename,nclass,mapoftclass,mode)#获取静态情况的边集合
    return  env(nclass,path,couplematrix,mapoftclass,setofedge)#返回环境env

#运行
def run(filename,osfpath,psfpath,stri,mode=0,Tt=1000):

    tsgenv = info2env(filename, mode)
    # 加载环境
    s_dim = tsgenv.getNClass()  # 状态空间
    a_dim = tsgenv.getNClass()  # 动作空间
    ddpg = DDPG(a_dim, s_dim)

    # 产生每次训练存储order的csv文件
    csvpath = osfpath + "/order_" + stri + ".csv"
    if not os.path.exists(csvpath):
        row = ["Order", "Cost", "FakeCost", "Generic Stubs", "Sepcific Stubs", "Number of Dependence",
               "Number of Attribute Dependence", "Number of Method Dependence", "Time of Appearance", "info about GS",
               "info about SS"]
        csvFile = open(csvpath, 'a+', newline='')
        csvwriter = csv.writer(csvFile)
        csvwriter.writerow(row)
        csvFile.close()

    episode_i = []
    costs = []

    for i in tqdm(range(10000)):
        batch_obs = []
        batch_trans = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []
        i_episode_cost = []
        T = 3000

        min_Cost = 10
        for t in range(1,T+1):
            state = tsgenv.reset()
            nonum = 1
            episode_rewards = []
            ifflag = False

            # 已选择动作集合初始化
            ChosenAction_list = []
            frame_idx = 0
            nonum = 0
            flag = True         # 是否有重复
            for j in range(tsgenv.getNClass()+1):
                action,log_prob,transfer = ddpg.get_action(state)
                batch_obs.append(state)
                batch_trans.append(transfer)
                batch_acts.append(action)
                ChosenAction_list.append(action)

                state,action,nextstate,reward,done,nonum,flag = tsgenv.step(action,nonum=nonum)

                episode_rewards.append(reward)
                batch_log_probs.append(log_prob)
                state = nextstate
                frame_idx += 1
                if done:
                    break
            if tsgenv.getrealcost() != 6:
                batch_lens.append(frame_idx)
                batch_rews.append(episode_rewards)
            if tsgenv.getrealcost() < min_Cost and len(ChosenAction_list) == tsgenv.getNClass() and flag == False:
                min_Cost = tsgenv.getrealcost()
                print(ChosenAction_list,min_Cost,"\t",t)
        episode_i.append(i)
        costs.append(min_Cost)
        if i % 10==0:
            plt.plot(episode_i, costs)
            plt.show()
        ddpg.update_model(batch_obs,batch_acts,batch_log_probs,batch_rews,batch_trans)


#创建文件夹
def preoption(filename):
    curpath = os.path.abspath(__file__)
    ppath=os.path.split(curpath)[0]
    date=datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    pltsavepath=ppath+"/plt"
    ordersavepath=ppath+"/order"
    print(pltsavepath)

    #创建目录
    if not os.path.exists(pltsavepath):
        os.mkdir(pltsavepath)

    if not os.path.exists(ordersavepath):
        os.mkdir(ordersavepath)

        #创建目录
    if not os.path.exists(pltsavepath+"/"+filename):
        os.mkdir(pltsavepath+"/"+filename)

    if not os.path.exists(ordersavepath+"/"+filename):
        os.mkdir(ordersavepath+"/"+filename)

    psfpath=pltsavepath+"/"+filename+"/"+date
    osfpath=ordersavepath+"/"+filename+"/"+date

    #创建与清除目录
    if not os.path.exists(psfpath):
        os.mkdir(psfpath)
    else:
        for pfile in os.listdir(psfpath):
            os.remove(psfpath+"/"+pfile)

    if not os.path.exists(osfpath):
        os.mkdir(osfpath)
    else:
        for pfile in os.listdir(osfpath):
            os.remove(osfpath+"/"+pfile)

    return osfpath,psfpath

if __name__ == '__main__':
    # 主程序
    '''
        |                   |
        |在这里改文件名、次数|
        V                   V
    '''
    filename = "SPM"  # 分析程序名称
    freq = 1  # 跑程序的次数
    mode = 0  # 1:动态 0:静态（默认动态）
    Tt = 10000  # 每次跑的时候的循环轮次（默认1000）
    print("-------------------", filename, "start!", "-------------------")
    osfpath, psfpath = preoption(filename)
    for index in range(0, freq):
         run(filename, osfpath, psfpath, str(index), mode, Tt)











