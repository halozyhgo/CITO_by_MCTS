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
from colour import Color

'''
    主程序
'''
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

'''
    这里只是声明参数，不用管
'''
beta_start = 0.4
beta_frames = 1000
epsilon_start = 1.0  #epsilon降低起点
epsilon_final = 0.01 #epsilon降低终点（无限趋近，但不会达到）
batch_size = 5   #记忆池单次抽取batch_size个样本用于训练
replay_buffer = NaivePrioritizedBuffer(5)   #记忆池初始化
epsilon_decay = 1                           #可以看epsilon做降低的速度
gamma      = 0.9                            #0.95左右效果比较好
T          = 15000                          #轮次，类多的话建议十万起步
min_value = -10                             #惩罚值
current_model = DuelingDQN(1, 1)
target_model  = DuelingDQN(1, 1)
optimizer = optim.Adam(current_model.parameters())


#beta的变化策略
beta_by_frame = lambda frame_idx: min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)
#epsilon的变化策略
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)
print(torch.cuda.is_available())

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

#更新target_model
def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())

#取样，计算loss
def compute_td_loss(batch_size, beta):
    state, action, reward, next_state, done, nonum,indices, weights = replay_buffer.sample(batch_size)

    state      = Variable(torch.FloatTensor(np.int64(state)))
    next_state = Variable(torch.FloatTensor(np.int64(next_state)))
    action     = Variable(torch.LongTensor(action))
    reward     = Variable(torch.FloatTensor(reward))
    done       = Variable(torch.FloatTensor(done))
    nonum      = Variable(torch.LongTensor(nonum))
    weights    = Variable(torch.FloatTensor(weights))

    q_values      = current_model(state)
    next_q_values = current_model(next_state)
    next_q_state_values = target_model(next_state) 

    q_value       = q_values.gather(1, action.unsqueeze(1)).squeeze(1) 
    next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)

    expected_q_value = reward + gamma * next_q_value * (1 - done)
    
    loss  = (q_value - expected_q_value.detach()).pow(2) * weights
    prios = loss + 1e-5
    loss  = loss.mean()
        
    optimizer.zero_grad()
    loss.backward()
    replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
    optimizer.step()
    
    return loss

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

#获取关于测试桩详细信息的字符串
def getstubstr(lis):
    stubstr="["
    for s in lis[:-1]:
        stubstr=stubstr+s+","
    stubstr=stubstr+lis[-1:][0]+"]"
    return stubstr

#将数据编程字符串，方便存储
def data2str(minorder,mincost,fakecost,needgs,needss,deps,attrdeps,methdeps,appear_time,gstub,sstub):
    order="["
    for num in minorder[:-1]:
        order=order+str(num)+","
    order=order+str((minorder[-1:])[0])+"]"
    c=float("%0.4f"%(mincost))
    fc=float("%0.4f"%(fakecost))
    gs=needgs
    ss=needss
    apt="%.0fms" % (appear_time)
    gs_str=getstubstr(gstub)
    ss_str=getstubstr(sstub)
    return [order,c,fc,gs,ss,deps,attrdeps,methdeps,apt,gs_str,ss_str]

#存储绘图数据
def pltdatatofile(x,y,name,psfpath):
    csvpath=psfpath+"/"+name+".csv"
    csvFile=open(csvpath,'a+',newline='') 
    csvwriter=csv.writer(csvFile)
    for index,i in enumerate(x):
        row=[i,y[index]]
        csvwriter.writerow(row)
    csvFile.close() 

#运行
def run(filename,osfpath,psfpath,stri,mode=0,Tt=1000):
    #产生每次训练存储order的csv文件
    csvpath=osfpath+"/order_"+stri+".csv"
    if not os.path.exists(csvpath):
        row=["Order","Cost","FakeCost","Generic Stubs","Sepcific Stubs","Number of Dependence","Number of Attribute Dependence","Number of Method Dependence","Time of Appearance","info about GS","info about SS"]
        csvFile=open(csvpath,'a+',newline='')
        csvwriter=csv.writer(csvFile)
        csvwriter.writerow(row)
        csvFile.close()

    #导入excel文件数据入环境
    tsgenv=info2env(filename,mode)

    '''
        |             |
        |在这里改参数|
        V             V
    '''
    global batch_size,replay_buffer,gamma,T,min_value,epsilon_decay,current_model,target_model,optimizer,beta_frames,\
            beta_start,beta_frames,epsilon_start,epsilon_final

    beta_start = 0.4
    beta_frames = 1000
    epsilon_start = 1.0  #epsilon降低起点
    epsilon_final = 0.001 #epsilon降低终点（无限趋近，但不会达到）    
    gamma      = 0.8        #0.95左右效果比较好
    T          = Tt     #轮次
    min_value = -150000    #惩罚值
    beta_frames = 1000    #beta提升速度
    batch_size = 64     #抽样数目（用于训练）
    replay_buffer = NaivePrioritizedBuffer(100000)   #记忆池初始化
    epsilon_decay = 1000*tsgenv.getNClass() #可以看epsilon做降低的速度

    current_model = DuelingDQN(tsgenv.getNClass(), tsgenv.getNClass())
    target_model  = DuelingDQN(tsgenv.getNClass(), tsgenv.getNClass())
    optimizer = optim.Adam(current_model.parameters())

    nclass=tsgenv.getNClass()
    if USE_CUDA:
        current_model = current_model.cuda()
        target_model  = target_model.cuda()

    mincost=sys.maxsize
    minorder=[]
    needgs=0
    needss=0

    betas=[]
    epsilons=[]
    losses = []
    all_rewards = []
    tlossindex=[]
    allflis=[]
    eposide_reward=0
    frame_idx=0

    stm_order=[]
    stm_cost=[]

    relit=[]
    fakelit=[]

    maxlenlist=[]
    orderlenlist=[]
    tindex=[]
    losses_idx=[]

    maxlen=-1

    #将动作集合放入totalactset
    totalactset=set()
    for i in range(0,tsgenv.getNClass()):
        totalactset.add(i)
    
    #训练T轮
    plt.ion() #实时绘制
    #print(('     ')+"  进度  "+('     ')+" frame_idx "+('     ')+" epsilon "+('     ')+" maxlen "+('     ')+" mincost "+('     ')+"  len  "+"\r")
    
    time_start = time.clock()
    plt.ion()
    for t in tqdm(range(1,T+1)):
        #环境初始化
        state=tsgenv.reset()
        nonum=1
        round_reward=0
        ifflag=False
        #已选动作集合初始化
        chosenactset=set()
        while(True):
            #确定epsilon的值
            epsilon=epsilon_by_frame(frame_idx)
            #预测动作action
            action = current_model.act(state,epsilon,tsgenv.getNClass())
            #env.step(),类同java里的那个step()步骤
            state,action,nextstate,reward,done,nonum,flag=tsgenv.step(action,nonum)
            #记忆池存储
            replay_buffer.push(state, action, reward, nextstate, done, nonum, frame_idx)
            #更新state,作为下次的初始状态
            state=nextstate.copy()
            #累加reward,用于画图
            round_reward=round_reward+reward
            #如果记忆池大小大于抽样的数目，就开始计算loss
            if len(replay_buffer) > batch_size:
                beta = beta_by_frame(frame_idx)
                loss = compute_td_loss(batch_size, beta)
                losses.append(loss.item())
                losses_idx.append(frame_idx)
                if done:
                    betas.append(beta)
                    tlossindex.append(frame_idx)

            if frame_idx%100==0:
                update_target(current_model, target_model)

            #allflis.append(frame_idx)
            if done:
                epsilons.append(epsilon)
            frame_idx=frame_idx+1

            #这轮结束了
            if done:
                ifflag=flag
                #记录每轮的代价，用于画图
                if flag:
                    relit.append(-1)
                    fakelit.append(-1)
                else:
                    relit.append(tsgenv.getrealcost())
                    fakelit.append(tsgenv.getcost())

                #更新序列最长长度
                if len(tsgenv.getorder())>maxlen and flag:
                    maxlen=len(tsgenv.getorder())-1
                elif not flag:
                    maxlen=nclass

                #更新代价最小的符合要求的序列
                if len(tsgenv.getorder())==tsgenv.getNClass() and ((mincost>tsgenv.getrealcost())or(abs(mincost-tsgenv.getrealcost())<0.000001))and (not flag) :
                    #更新最小序列
                    mincost=tsgenv.getrealcost()
                    fakecost=tsgenv.getcost()
                    minorder=tsgenv.getorder().copy()
                    needgs=tsgenv.getgenstub()
                    needss=tsgenv.getspstub()
                    deps=tsgenv.getnumsofdeps()
                    min_cost_time= time.clock()
                    appear_time=(min_cost_time - time_start)*1000
                    methdeps=tsgenv.getnumsofmethdeps()
                    attrdeps=tsgenv.getnumsofattrdeps()
                    gstub=tsgenv.get_gstub()
                    sstub=tsgenv.get_sstub()
                    #录入csv
                    row=data2str(minorder,mincost,fakecost,needgs,needss,deps,attrdeps,methdeps,appear_time,gstub,sstub)
                    csvFile=open(csvpath,'a+',newline='')
                    csvwriter=csv.writer(csvFile)
                    csvwriter.writerow(row)
                    csvFile.close()
                #输出的为：帧数，最小代价，类总数，最长长度，当前轮的序列代价
                #sys.stdout.write(('     ')+"  进度  "+('     ')+" frame_idx "+('     ')+" epsilon "+('     ')+" maxlen "+('     ')+" mincost "+('     ')+"  len  "+"\r")
                #sys.stdout.write(('       ')+("%.4f"%((t+1)/T))+('       ')+str(frame_idx)+('          ')+("%.4f"%(epsilon))+('          ')+str(maxlen)+('       ')+str(mincost)+('       ')+str(len(tsgenv.getorder()))+"\r")
                #sys.stdout.flush()
                print("第"+(str(t+1))+"轮训练结果： ",frame_idx,epsilon,flag,"%.4f"%(mincost),tsgenv.getNClass(),maxlen,len(tsgenv.getorder()),"cost=%0.4f"%(tsgenv.getcost()),tsgenv.getorder())
                #print("第"+(str(t+1))+"轮训练结果： ",frame_idx,epsilon,flag,"%.4f"%(mincost),tsgenv.getNClass(),maxlen,len(tsgenv.getorder()),"cost=%0.4f"%(tsgenv.getcost()))
                #sys.stdout.write(('进度：')+("%.4f"%((t+1)/T))+" "+"%.4f"%(mincost)+'\r')
                #sys.stdout.flush()
                break
        if t%10000==0:
            t_time = time.clock()
            appear_time = (t_time - time_start) * 1000
            row =["#"+str(t) ,"%.0fms" % (appear_time)]
            csvFile = open(csvpath, 'a+', newline='')
            csvwriter = csv.writer(csvFile)
            csvwriter.writerow(row)
            csvFile.close()
        #保存当前得到的最长序列的长度，用于画图
        maxlenlist.append(maxlen)
        #保存每轮序列的长度，用于画图
        if flag:
            orderlenlist.append(len(tsgenv.getorder())-1)
        else:
            orderlenlist.append(len(tsgenv.getorder()))
        #保存每轮累计reward
        all_rewards.append(round_reward)
        #保存轮数，用于画图
        tindex.append(t)
        
        #实时绘图，每5轮画张图
        if t % 5  == 0:
            plt.figure(1)
            plt.cla()
            plt.xlabel("t")
            plt.ylabel("orderlen")
            plt.plot(tindex,orderlenlist,color='dodgerblue',alpha=0.7)
            plt.plot(tindex,maxlenlist,color='b')
            plt.pause(0.1)
        
    plt.ioff()
    plt.clf()
    plt.close()
    
    time_end = time.clock()
    #每轮真cost的情况
    plt.figure(2)
    #plt.plot(tindex,relit,color='dodgerblue')
    plt.bar(tindex,relit,width=1.0,color='dodgerblue',alpha=0.7)
    plt.xlabel("t")
    plt.ylabel("the Real Cost of Sequence")
    plt.title("Diagram of the change of real cost of sequence per round")
    plt.savefig(psfpath+"/"+filename+"_t_realcost_"+stri+".png")
    plt.clf()
    plt.close()
    pltdatatofile(tindex,relit,filename+"_t_realcost_"+stri,psfpath)

    #每轮伪cost的情况
    plt.figure(3)
    plt.xlabel("t")
    plt.ylabel("the Fake Cost of Sequence")
    plt.title("Diagram of the change of fake cost of sequence per round")
    #plt.plot(tindex,relit,color='dodgerblue')
    plt.bar(tindex,fakelit,width=1.0,color='dodgerblue',alpha=0.7)
    plt.savefig(psfpath+"/"+filename+"_t_fakecost_"+stri+".png")
    plt.clf()
    plt.close()
    pltdatatofile(tindex,fakelit,filename+"_t_fakecost_"+stri,psfpath)

    #每轮累计奖励情况
    plt.figure(4)
    plt.xlabel("t")
    plt.ylabel("Accumulated Rewards")
    plt.title("Diagram of the change of accumulated rewards per round")
    plt.plot(tindex,all_rewards, lw=1.0,color='dodgerblue',alpha=0.7)
    #plt.plot(tindex,all_rewards,color='dodgerblue')
    plt.savefig(psfpath+"/"+filename+"_t_reward_"+stri+".png")
    plt.clf()
    plt.close()
    pltdatatofile(tindex,all_rewards,filename+"_t_reward_"+stri,psfpath)

    #每帧loss变化情况
    plt.figure(5)
    plt.xlabel("Frame")
    plt.ylabel("Loss")
    plt.title("Diagram of the change of loss per frame")
    #plt.bar(tlossindex,losses,width=1.0,color='dodgerblue',alpha=0.7)
    plt.plot(losses_idx,losses,color='dodgerblue')
    plt.savefig(psfpath+"/"+filename+"_frame_loss_"+stri+".png")
    plt.clf()
    plt.close()
    pltdatatofile(losses_idx,losses,filename+"_frame_loss_"+stri,psfpath)

    plt.figure(6)
    plt.xlabel("t")
    plt.ylabel("Epsilon")
    plt.title("Diagram of the change of epsilon per round")
    plt.plot(tindex,epsilons,color='dodgerblue')
    plt.savefig(psfpath+"/"+filename+"_t_epsilon_"+stri+".png")
    plt.clf()
    plt.close()
    pltdatatofile(tindex,epsilons,filename+"_t_epsilon_"+stri,psfpath)

    plt.figure(7)
    plt.xlabel("t")
    plt.ylabel("Beta")
    plt.title("Diagram of the change of beta per round")
    plt.plot(tlossindex,betas,color='dodgerblue')
    plt.savefig(psfpath+"/frame_beta_"+stri+".png")
    plt.clf()
    plt.close()
    pltdatatofile(tlossindex,betas,"frame_beta_"+stri,psfpath)

    plt.figure(8)
    plt.xlabel("t")
    plt.ylabel("the Length of Order")
    plt.title("Diagram of the change of length of order per round")
    plt.plot(tindex,orderlenlist,color="dodgerblue",alpha=0.7)
    plt.plot(tindex,maxlenlist,color='b',linestyle="--")
    plt.savefig(psfpath+"/"+filename+"_t_orderlen_"+stri+".png")
    plt.clf()
    plt.close()
    pltdatatofile(tindex,orderlenlist,filename+"_t_orderlen_"+stri+"_1",psfpath)
    pltdatatofile(tindex,maxlenlist,filename+"_t_orderlen_"+stri+"_2",psfpath)

    

    return minorder,mincost,needgs,needss,time_start,time_end

if __name__ == '__main__':
    #主程序
    '''
        |                   |
        |在这里改文件名、次数|
        V                   V
    '''
    filename="test"    #分析程序名称
    freq=1  #跑程序的次数
    mode=0  #1:动态 0:静态（默认动态）
    Tt=1000 #每次跑的时候的循环轮次（默认1000）
    print("-------------------",filename,"start!","-------------------")
    osfpath,psfpath=preoption(filename)
    for index in range(0,freq):
        minorder,mincost,needgs,needss,time_start,time_end=run(filename,osfpath,psfpath,str(index),mode,Tt)
        
        print(">>No.",index," time:","%.0fms" % ((time_end - time_start)*1000))
        print("  <>lowest order:",minorder)
        print("  <>cost:        ","%.4f"%(mincost))
        print("  <>GS:          ",needgs)
        print("  <>SS:          ",needss)


        


   

