import random


class DecisionTreeNode:
    def __init__(self):
        self.name = None
        self.values = None
        self.leafNode = None
        self.dataSet = None
        self.children = {}

    def setParameters(self, name, values, data=None):
        self.name = name
        self.values = values
        self.dataSet = data
        for val in self.values:
            self.children[val] = None

    def setLeafClass(self, label):
        self.leafNode = label

    def addChild(self, val, node):
        self.children[val] = node


    ''' sometimes return random labels in case of multiple options '''
    def getClass(self):
        if type(self.leafNode) == list: return  random.choice(self.leafNode)
        else: return self.leafNode

    def __repr__(self):
        if self.leafNode != None:
            return "leaf_node_class = " + repr(self.leafNode)
        return 'Node('+self.name + ')'




