import matplotlib.pyplot as plt
import csv
from operator import itemgetter
import math
'''
    该程序用于绘制与beta、epsilon相关的图
'''
path="plt\\"
filename="test"
tstr="2021_03_26_19_32_11\\"
x1=[]
y1=[]
x2=[]
y2=[]

filepath1=path+filename+"\\"+tstr+filename+"_t_epsilon_0.csv"
filepath2=path+filename+"\\"+tstr+"frame_beta_0.csv"

#beta的变化策略
beta_by_frame = lambda frame_idx: min(1.0, 0.4 + frame_idx * (1 - 0.4) / 1000)
#epsilon的变化策略
epsilon_by_frame = lambda frame_idx: 0.001 + (1.0 - 0.001) * math.exp(-1. * frame_idx / 8000)

f1=open(filepath1,'r')
f2=open(filepath2,'r')
reader1=csv.reader(f1)
reader2=csv.reader(f2)

for i in range(0,30001):
    x1.append(i)
    y1.append(beta_by_frame(i))

for i in range(0,30001):
    x2.append(i)
    y2.append(epsilon_by_frame(i))


plt.figure(1)
plt.plot(x1,y1,color="c",label='beta',linewidth=2)
plt.plot(x2,y2,color="orange",label='epsilon',linewidth=2)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Figure of changes in epsilon and beta")
plt.legend(fontsize=12)
plt.show()