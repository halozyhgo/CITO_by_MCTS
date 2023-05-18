import sys

from TClass import tclass
from TEdge import tedge
from rtype import rtype

min_value = -150000

class env(object):
    def __init__(self,nclass,path,couplematrix,mapofclasses,setofedges):
        self.nclass=nclass#类数
        self.path=path#路径连通情况
        self.cplx=couplematrix#耦合矩阵
        self.mapofclasses=mapofclasses#类与类序号映射
        self.setofedges=setofedges#边集合
        self.costmatrix=[[0.0]*nclass for _ in range(nclass)]#花费矩阵
        self.rcostmatrix=[[0.0]*nclass for _ in range(nclass)]#真花费矩阵
        self.genericstubs=0#通用测试桩数
        self.specificstubs=0#特殊测试桩数
        self.GS=[False]*nclass #有无建立通用测试桩
        self.numsofdeps=0#依赖总数目
        self.select=[False]*nclass#是否选择了
        self.cost=0.0#
        self.realcost=0.0
        self.c=10000
        self.flag = False

        self.numsofmethdeps=0
        self.numsofattrdeps=0

        self.mincost=sys.maxsize
        self.minSS=1000
        self.minGS=1000
        self.minDeps=0

        self.max=1500       # 奖励其实是 15 0000 后面有乘 100

        self.order=[]
        self.gstub=[]
        self.sstub=[]

        self.curstate=[-1]*nclass

        self.initCostMatrix()

    def getNClass(self):
        return self.nclass

    def initCostMatrix(self):
        for index,pedge in enumerate(self.setofedges):
            fromindex=pedge.getfromindex()
            toindex=pedge.gettoindex()
            self.costmatrix[fromindex][toindex]=self.cplx[fromindex][toindex]
            self.rcostmatrix[fromindex][toindex]=self.cplx[fromindex][toindex]
            if pedge.gettype()==rtype.AG or pedge.gettype()==rtype.I:
                self.costmatrix[fromindex][toindex]=5*self.cplx[fromindex][toindex]
        #print("cmt:",'\n',self.costmatrix)
        #print("rcmt:",'\n',self.rcostmatrix)
        return 

    def getorder(self):
        return self.order

    def get_state_result(self):
        done = True
        if self.curstate.count(-1) != 0 and self.flag==False:
            done = False
            return done,None
        else:
            return done,-self.getcost()

    #计算奖励
    def calculatereward(self,action,flag,nonum):
        if flag:#有重复的情况
            #return (self.nclass-nonum+1)*min_value 
            return min_value

        reward=0.0
        #print(nonum,self.nclass)
        if nonum<self.nclass:#符合要求，但是还没选完
            #print(">>>>",nonum,":")
            profit=self.calculateSpecificStub(action)
            reward=self.c*(profit)

        if nonum==self.nclass and self.realcost<=self.mincost:#符合要求，已经选完，是否代价为最小的
            self.mincost=self.realcost
            self.minGS=self.genericstubs
            self.minSS=self.specificstubs
            self.mindeps=self.numsofdeps

            return 100*self.max 
        else:
            return reward
    #重置环境
    def reset(self):
        self.order=[]
        self.gstub=[]       # 通用测试桩重置
        self.sstub=[]       # 特定测试桩重置
        self.curstate=[-1.0]*self.nclass
        self.select=[False]*self.nclass
        self.GS=[False]*self.nclass
        self.cost=0
        self.realcost=0

        self.specificstubs=0
        self.genericstubs=0
        self.numsofdeps=0
        self.numsofmethdeps=0
        self.numsofattrdeps=0

        # 改变状态的表示1
        # self.curstate.append(self.getrealcost())

        return self.curstate
    #环境交互
    def step(self,action):
        done=False
        nextstate=self.curstate.copy()
        self.flag=False
        #更改状态
        for index,pselect in enumerate(self.select):
            if pselect and (index==action):#有重复
                nextstate[index]=-10
                self.flag=True
                break
            if ( not pselect ) and (index==action):#无重复
                nextstate[index]= 1          # 将已经选择的类下标改为 1
                # nextstate[index]=nonum
                self.flag=False
                break
        #计算奖励       
        reward=self.calculatereward(action,self.flag,nextstate.count(1))

        self.fresh(action)#更新
        
        if len(self.order)==self.nclass or self.flag:
            done=True
        
        state=self.curstate.copy()
        self.curstate=nextstate.copy()
        #返回当前状态，动作，下一个状态，奖励，是否完成标志，下一个动作的次序，是否重复标志
        # state, reward, done, _
        return state,action,nextstate,reward,done,self.flag
        # return nextstate,reward,done,flag,nonum

    #更新order,select
    def fresh(self,action):
        self.order.append(action)
        self.select[action]=True

    #计算特殊测试桩，以及奖励计算
    def calculateSpecificStub(self,action):
        totalcost=0.0
        realtotalcost=0.0
        totalprofit=0.0
        source=tclass(-1,"null",dict(),dict())
        target=tclass(-1,"null",dict(),dict())
        act=action
        for i in range(self.nclass):
            #print("###",i,act)
            if self.path[act][i]>0:
                if self.select[i]:
                    continue
                totalcost=totalcost+self.costmatrix[act][i]
                realtotalcost=realtotalcost+self.rcostmatrix[act][i]
                if not self.GS[i]:
                    self.GS[i]=True
                    self.genericstubs=self.genericstubs+1
                    self.gstub.append(str(i)+":"+(self.mapofclasses.get(i)).getcname()) # 通过索引获得名字
                self.specificstubs=self.specificstubs+1
                self.sstub.append(str(i)+"("+str(act)+"):"+self.mapofclasses.get(i).getcname()+"("+self.mapofclasses.get(act).getcname()+")")
                source=self.mapofclasses.get(act)
                target=self.mapofclasses.get(i)
                #print(type(act),type(source.getattrdeps()),type(source.getattrdeps().get(target.getcname())),source.getattrdeps().get(target.getcname()))
                if target.getcname() in source.getattrdeps():
                    self.numsofdeps=self.numsofdeps+int(source.getattrdeps().get(target.getcname()))
                    self.numsofattrdeps=self.numsofattrdeps+int(source.getattrdeps().get(target.getcname()))
                if target.getcname() in source.getmethdeps():
                    self.numsofdeps=self.numsofdeps+int(source.getmethdeps().get(target.getcname()))
                    self.numsofmethdeps=self.numsofmethdeps+int(source.getmethdeps().get(target.getcname()))
            if self.path[i][act]>0 and (not self.select[i]):
                totalprofit=totalprofit+self.costmatrix[i][act]
                #realtotalcost=realtotalcost+self.rcostmatrix[i][act]
        self.cost=self.cost+totalcost
        self.realcost=self.realcost+realtotalcost

        return totalprofit-totalcost

    def get_available_actions(self,state):
        """
        这里的state已经转化成了 list : [-1,2,....]
        :param state: 当前的状态
        :return: 可以执行的动作们
        """
        available_actions = []
        for i,item in enumerate(state):
            if item == -1.0:
                available_actions.append(i)
        return available_actions

    #获取order
    def getorder(self):
        return self.order

    #计算代价
    def justcalcost(self,action,nonum,pstate):
        done=False
        nextstate=pstate.copy()
        flag=False
        #更改状态
        for index,pselect in enumerate(self.select):
            if pselect and (index==action):#有重复
                nextstate[index]=-10
                flag=True
                break
            if ( not pselect ) and (index==action):#无重复
                nextstate[index]=nonum
                flag=False
                break

        reward=self.calculatereward(action,flag,nonum)

        self.fresh(action)#更新
        
        if len(self.order)==self.nclass or reward==min_value:
            done=True
    
        self.curstate=nextstate.copy()
        #nonum=nonum+1
        #返回“是否完成”标志、“是否有重复”标志
        return done,flag
    
    #获取伪代价（因为有的复杂度乘5了，这样计算出来的复杂度偏大）
    def getcost(self):
        return self.cost

    #获取真代价（利用原本复杂度计算的）
    def getrealcost(self):
        return self.realcost
    
    #特定测试装数目
    def getspstub(self):
        return self.specificstubs
    
    #通用测试桩数目
    def getgenstub(self):
        return self.genericstubs

    #依赖总数目
    def getnumsofdeps(self):
        return self.numsofdeps

    #方法依赖数目
    def getnumsofmethdeps(self):
        return self.numsofmethdeps
    
    #属性依赖数目
    def getnumsofattrdeps(self):
        return self.numsofattrdeps

    #特定测试桩设定情况
    def get_gstub(self):
        return self.gstub
    
    #通用测试桩设定情况
    def get_sstub(self):
        return self.sstub

