class A(object):

    @classmethod
    def lappa(cls):
        a = [1, 2]
        print ("A lappa")
        return a

    def __init__(self, params=None):
        if params is None:
            print ("A init")
            self.params = self.lappa()
        else:
            self.params = params

class B(A):

    @classmethod
    def lappa(cls):
        print ("B lappa")
        b = super(B, cls).lappa()
        b += ["ddsfkjhsdsfgsdkf"]
        return b

    def __init__(self, params=None):
        super(B, self).__init__(params)


a = B()
print (a.params)
