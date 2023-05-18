import csv
import re
from operator import itemgetter

from TClass import tclass
from TEdge import tedge

'''
    该程序用于读取excel表格
'''

path=".\\infodata\\" #相对路径

#获取类个数
def getNClass(filename):
    f=open(path+filename+"/CId_Name.csv")
    NClass=sum(1 for line in f)-1
    f.close()
    return NClass

#获取动态分析中的path(可能一个from->to有多重关系，所以此处只能用于判断连通不连通)
def getDynamicPath(filename,n):
    f=open(path+filename+"/deps_type.csv",'r')
    reader=csv.reader(f)
    sortedList = sorted(reader, key=itemgetter(0))
    sortedList.remove(sortedList[reader.line_num-1])
    dypath=[[0]*n for _ in range(n)]
    for index,row in enumerate(sortedList):
        fromindex=int(row[0])
        toindex=int(row[1])
        typenum=int(row[2])
        dypath[fromindex][toindex]=typenum
    f.close()
    return dypath

#获取静态分析中的path(推荐此处只用于判断连通不连通)
def getStaticPath(filename,n):
    f=open(path+filename+"/deps_type.csv",'r')
    reader=csv.reader(f)
    sortedList = sorted(reader, key=itemgetter(0))
    sortedList.remove(sortedList[reader.line_num-1])
    stpath=[[0]*n for _ in range(n)]
    for index,row in enumerate(sortedList):
        fromindex=int(row[0])
        toindex=int(row[1])
        typenum=int(row[2])
        if typenum!=4:
            stpath[fromindex][toindex]=typenum
    f.close()
    return stpath

#获取id与名字的字典
def getIdNamedict(filename):
    Id=[]
    Name=[]
    with open(path+filename+"/CId_Name.csv",'r') as f:
        reader=csv.reader(f)
        sortedList = sorted(reader, key=itemgetter(0))
        sortedList.remove(sortedList[reader.line_num-1])
        for index,row in enumerate(sortedList):
            Id.append(int(row[0]))
            Name.append(str(row[1]))
        f.close()
    dic=dict(zip(Id,Name))
    return dic

#获得类间耦合度矩阵
def getCoupleMatrix(filename,n,mode=1):
    CostMatrix=[[0]*n for _ in range(n)]
    Name=[]
    mapoftclass=getMAPofTClass(filename)
    setoftedge=getSETofTEdge(filename,n,mapoftclass,mode)
    with open(path+filename+"/Couple_List.csv",'r') as f:
        reader=csv.reader(f)
        for index,row in enumerate(reader):
            if index:
                CostMatrix[int(row[0])][int(row[1])]=float(row[2])
        f.close()
    return CostMatrix    

#获得类编号为classindex的类的属性相关的类信息
def getperTClassAttrdeps(filename,classindex):
    Name=[]
    Attrnum=[]
    with open(path+filename+"/Attr_Method_deps.csv",'r') as f:
        reader=csv.reader(f)
        sortedList = sorted(reader, key=itemgetter(0))
        sortedList.remove(sortedList[reader.line_num-1])
        for row in sortedList:
            if int(row[0])==classindex:
                attrdeps=row[2]
                if attrdeps=="null":
                    break
                splattr=re.split(r'{|}|,',attrdeps)
                for index,part in enumerate(splattr):
                    if index!=1 and len(part)!=0:
                        splpart=part.split(':')
                        Name.append(splpart[0])
                        Attrnum.append(splpart[1])
    dic=dict(zip(Name,Attrnum))
    return dic

#获得类编号为classindex的类的属性相关的类信息
def getperTClassMethoddeps(filename,classindex):
    Name=[]
    Methodnum=[]
    with open(path+filename+"/Attr_Method_deps.csv",'r') as f:
        reader=csv.reader(f)
        sortedList = sorted(reader, key=itemgetter(0))
        sortedList.remove(sortedList[reader.line_num-1])
        for row in sortedList:
            if int(row[0])==classindex:
                methdeps=row[3]
                if methdeps=="null":
                    break
                splattr=re.split(r'{|}|,',methdeps)
                for index,part in enumerate(splattr):
                    if index!=1 and len(part)!=0:
                        splpart=part.split(':')
                        Name.append(splpart[0])
                        Methodnum.append(splpart[1])
    dic=dict(zip(Name,Methodnum))
    return dic

#获取TClass映射
def getMAPofTClass(filename):
    listoftclass=[]
    Id=[]
    dictofnameid=getIdNamedict(filename)
    for idnum,name in dictofnameid.items():
        attrdeps=getperTClassAttrdeps(filename,int(idnum))
        methdeps=getperTClassMethoddeps(filename,int(idnum))
        pTClass=tclass(idnum,name,attrdeps,methdeps)
        Id.append(int(idnum))
        listoftclass.append(pTClass)
    mapoftclass=dict(zip(Id,listoftclass))
    return mapoftclass

#获取TEdge集合
def getSETofTEdge(filename,n,mapoftclass,mode=1):
    listoftedge=[]
    f=open(path+filename+"/deps_type.csv",'r')
    reader=csv.reader(f)
    sortedList = sorted(reader, key=itemgetter(0))
    sortedList.remove(sortedList[reader.line_num-1])
    if mode:#动态分析(包含Dy)
        for index,row in enumerate(sortedList):
            fromindex=int(row[0])
            toindex=int(row[1])
            typenum=int(row[2])
            ptedge=tedge(mapoftclass.get(fromindex),mapoftclass.get(toindex),typenum)
            listoftedge.append(ptedge)
    else:#静态分析(去除Dy)
        for index,row in enumerate(sortedList):
            fromindex=int(row[0])
            toindex=int(row[1])
            typenum=int(row[2])
            if typenum!=4:
                ptedge=tedge(mapoftclass.get(fromindex),mapoftclass.get(toindex),typenum)
                listoftedge.append(ptedge)
    f.close()
    return listoftedge            

def getimportance(filename,mode=1):
    Id=[]
    Impo=[]
    with open(path+filename+"/CId_importance.csv",'r') as f:
        reader=csv.reader(f)
        sortedList = sorted(reader, key=itemgetter(0))
        sortedList.remove(sortedList[reader.line_num-1])
        for index,row in enumerate(sortedList):
            Id.append(int(row[0]))
            Impo.append(str(row[1]))
        f.close()
    dic=dict(zip(Id,Impo))
    return dic    
'''
#用于测试函数是否正确
NClass=getNClass("ATM")
getIdNamedict("ATM")
getCoupleMatrix("ATM",NClass)
getperTClassAttrdeps("ATM",7)
getperTClassMethoddeps("ATM",7)
getDynamicPath("ATM",NClass)
print(getStaticPath("ATM",NClass))
mapoftclass=getMAPofTClass("ATM")
mapoftclass.get(1).printinfo()#只是测试getMAPofTClass
for i in range(0,NClass):
    print("****",i)
    mapoftclass.get(i).printinfo()
getSETofTEdge("ATM",NClass,mapoftclass,1)#动态分析情况
getSETofTEdge("ATM",NClass,mapoftclass,0)#静态分析情况
'''
