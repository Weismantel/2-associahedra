import numpy as np
from queue import Queue
import time

class PosetNode:
    #Simple poset structure. Every element stores all their children
    def __init__(self, val, codim = 0):
        self.val = val
        self.codim = codim
        self.children = []
        self.parents = []

    def add_child(self, nodes):
        self.children = self.children + nodes
        for node in nodes:
            node.parents.append(self)
    
    def __str__(self):
            return str(self.val)
    
    def __lt__(a,b):
        #Compares the strings of the 1-bracketings using the Unicode table
        if str(a.val) < str(b.val):
            return True
        elif str(a.val) > str(b.val):
            return False
        else:
            return None
        

class Kr():
    def __init__(self, r):
        self.r = r
        r, n = Kr.build(r)
        self.root = r
        self.nodes = n
        self.Gamma = None
        self.Psi = None
        self.dim = self.r - 2
    
    def choose_iter(elements, length):
        for i in range(len(elements)):
            if length == 1:
                yield (elements[i],)
            else:
                for next in Kr.choose_iter(elements[i+1:], length-1):
                    yield (elements[i],) + next

    def choose(l, k):
        #Calles the function choose_iter to return all the possibilities to pick k elements out of l as a list.
        return list(Kr.choose_iter(l, k))

    def make_list_brackets(l, br):
        #Takes list of pairs of integeres as entry and makes a bracket around the elemtents with index of the integers.
        res = []
        for b in br:
            bracket = [l[b[0]:b[1]+1]]
            res.append(l[:b[0]] + bracket + l[b[1]+1:])

        return res

    def make_all_brackets(li):
        #Tackes a list as an input and makes all possible degenerations on them
        res = []
        l = len(li)
        #Starts with the outer bracket and makes a new in there, if there are more then three elements in there
        if l >= 3:
            poss = Kr.choose(np.arange(l), 2)
            poss.remove((0,l-1))
            res = res + Kr.make_list_brackets(li, poss)
        
        #If there are lists contained in the list recursivly make all possible brackets in the list
        for i in range(l):
            if type(li[i]) == list:
                #Produce all the subbrackets subbr in the 1-bracket li[i]
                subbr = Kr.make_all_brackets(li[i])
                for j in range(len(subbr)):
                    #Embedd the subbrackets into the whole 1-bracketing
                    newbr = list(li)
                    newbr[i] = list(subbr[j])
                    res.append(newbr)
        
        return res

    def build(r):
        #Takes r as input and gives out the Associahedra Kr given by a poset structure.
        #Initialize dictionaries to store the faces of given codimension.
        #We call the faces as nodes of the poset.
        Nodes = [ {} for i in range(r-1)]
        #Initialize codimension 0 face, 'root'. 
        n = list(np.arange(r))
        root = PosetNode(n, 0)
        Nodes[0][str(root.val)] = root
        #We use a queue to iterate over the faces and recursivley make all children.
        Q = Queue()
        Q.put(root)
        while not Q.empty():
            #Take node out of the Queue and make all possible degenerations on it
            curnode = Q.get()
            newcodim = curnode.codim + 1
            childs = list(Kr.make_all_brackets(curnode.val))
            for child in childs:
                #Use the dictionary to check if the child was already created. For this use the string representation which is just given by str(child).
                if str(child) in Nodes[newcodim]:
                    #If child node is already in the node dictionary, set the excisting node as child of the current node 'curnode'. 
                    innode = Nodes[newcodim][str(child)]
                    curnode.add_child([innode])
                else:
                    #If not add the child as a node and put it in the Queue
                    newnode = PosetNode(child, newcodim)
                    curnode.add_child([newnode])
                    Nodes[newcodim][str(child)] = newnode
                    Q.put(newnode)

        return root, Nodes

    def get_number_nodes(self):
        sum = 0
        for elt in self.nodes: sum = sum + len(elt)
        return sum
    
    def f_vector(self):
        #Prints out the f-vector of the poset.
        return list(reversed([len(self.nodes[i]) for i in range(len(self.nodes))]))
                    
    def alternating(self):
        #Prints out the alternating some of the f-vector. It has to be 1, since the Associahedra is a polytope.
        f_vector = self.f_vector()
        result = 0
        for i in range(len(f_vector)):
            if i % 2 == 0:
                result += f_vector[i]
            else:
                result -= f_vector[i]
        return result
    
    
    

if __name__ == '__main__':
    
    start = time.time()
    K = Kr(8)
    end = time.time()
    timer = end-start
    print(f'Time: {timer}')
    print(f'f_vector: {K.f_vector()}, alternating sum: {K.alternating()}')


    K = Kr(5)
    for i in range(len(K.nodes)):
        print(f'Codimension {i} faces:')
        for key in K.nodes[i]:
            print(K.nodes[i][str(key)])

