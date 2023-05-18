#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : Administrator
# date   : 2018/6/28
from game import Game
from mcts import MCTS
from TSGenv import env
import LoadSootdata

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

if __name__ == '__main__':
    game_name = "test"
    game_cito = info2env(game_name,mode=0)
    game = Game()
    ai = MCTS()
    chosenActions = []
    while True:
        action = ai.take_action(game_cito)
        game_cito.step(action)
        chosenActions.append(action)
        print("###{0}选择第{1}个类###".format(ai, action))

        # 判断结果
        is_over, winner = game_cito.get_state_result()
        if is_over:
            print(game_cito.getcost())
            break


