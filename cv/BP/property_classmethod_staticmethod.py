class cal:
    cal_name = '计算器'
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def cal_sub(self):
        return self.x - self.y

    @property           #在cal_add函数前加上@property，使得该函数可直接调用，封装起来
    def cal_add(self):
        return self.x + self.y

    @classmethod        #在cal_info函数前加上@classmethon，则该函数变为类方法，该函数只能访问到类的数据属性，不能获取实例的数据属性
    def cal_info(cls):  #python自动传入位置参数cls就是类本身
        print('这是一个%s'%cls.cal_add)   #cls.cal_name调用类自己的数据属性

    @staticmethod        #静态方法 类或实例均可调用
    def cal_test(a,b,c): #改静态方法函数里不传入self 或 cls
        print(a,b,c)


c = cal(10,11)
print(c.cal_sub)
# print(c.cal_add)


# 如果不想通过实例来调用类的函数属性，而直接用类调用函数方法，则这就是类方法，通过内置装饰器@calssmethod
# cal.cal_info()



# @staticmethod 静态方法只是名义上归属类管理，但是不能使用类变量和实例变量，是类的工具包
# 放在函数前（该函数不传入self或者cls），所以不能访问类属性和实例属性
# c1 = cal(10,11)
# cal.cal_test(1,2,3)     #>>> 1 2 3    类调用
# c1.cal_test(1,2,3)      #>>> 1 2 3    实例调用