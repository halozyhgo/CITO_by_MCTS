'''
    TClass定义代码
'''
class tclass(object):
    def __init__(self,Id,Name,attrdeps,methdeps):
        self.id=Id
        self.name=Name
        self.attrdeps=attrdeps
        self.methdeps=methdeps

    #打印节点信息
    def printinfo(self):
        print("Id:",self.id)
        print("Name:",self.name)
        print("Attrbute Dependence:")
        for index,item in enumerate(self.attrdeps.items()):
            print("#"+str(index)," ",item)
        print("Method Dependence:")
        for index,item in enumerate(self.methdeps.items()):
            print("#"+str(index)," ",item)
    
    #获取类的名称
    def getcname(self):
        return self.name
    
    #获取属性依赖表,类型为字典
    def getattrdeps(self):
        return self.attrdeps
    
    #获取方法依赖表，类型为字典
    def getmethdeps(self):
        return self.methdeps
    
    #获取id
    def getid(self):
        return self.id
