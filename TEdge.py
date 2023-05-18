from TClass import tclass
'''
    TEdge的定义代码
'''
class tedge(object):
    def __init__(self,fromtclass,totclass,etype):
        self.fromtc=fromtclass
        self.totc=totclass
        self.type=etype
        self.fromindex=fromtclass.getid()
        self.toindex=totclass.getid()
    
    #获取出点的类
    def getfromclass(self):
        return self.fromtc
    
    #获取入点的类
    def gettoclass(self):
        return self.totc
    
    #获取边的依赖类型
    def gettype(self):
        return self.type

    #获取出点的类的序号
    def getfromindex(self):
        return self.fromindex
    
    #获取入点的类的序号
    def gettoindex(self):
        return self.toindex
