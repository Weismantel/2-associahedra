import numpy as np
from queue import Queue
from helper import *
import copy
from functools import cmp_to_key
import time


class PosetNode():
    #simple structure to store a poset.
    # Each node has it's one smaller and one bigger elements stored in children/parents

    def __init__(self, d, a, n, codim = 0):
        self.d = d
        self.a = a
        self.n = n
        self.codim = codim
        self.children = []
        self.parents = []

    def add_child(self, nodes):
        self.children = self.children + nodes
        for node in nodes:
            node.parents.append(self)

    def __str__(self):
        return string_representation(self)
        
    def __lt__(node1, node2):
        #Comparison function on to poset nodes, to use binary search.
        #The comparison function compares the isolated 1- & 2-brackets of the node 
        return compare_orders(node1, node2)


class Wn():
    def __init__(self, n):
        self.n = n
        self.marked_points = self.gen_marked_points()
        self.root = self.gen_root()
        self.nodes = self.build()
        self.dim = len(self.n) - 3
        for elt in self.n: self.dim = self.dim + elt

    def gen_marked_points(self):
        #Generates the marked points on the seams. The points are indexed by (i,j) \in {1,...,r} \times {1,...,n_i}
        points = [[] for i in range(len(self.n))]
        for i in range(len(self.n)):
            points[i] = [[(i, j)] for j in range(self.n[i])]
        return points
    
    def gen_root(self):
        #Generates the codimension 0 face, the 'root', of the poset.
        d = {}
        for j in range(len(self.marked_points)):
            d[str([j])] = [self.marked_points[j]]
        a = list(np.arange(len(self.n)))
        root = PosetNode(d, a, self.n)
        return root
    
    def get_partitions(marked_points):
        '''
        Input:  List of 2-brackets over 1-brackets
        Output: Partitions of the 2-brackets. The 2-brackets in one partition are made into a new 2-bracket.
                The method only returns partitions that are valid new 2-brackets.
        '''
        final_part = []
        if marked_points == [] or marked_points == [[]]:
            return []
        #Get lists in the length of the number of marked points over each 1-bracket + 1. 
        #The +1 is for the '0' telling us that we pick no marked point over that specific 1-bracket.
        lists = [list(np.arange(len(marked_points[i])+1)) for i in range(len(marked_points))]

        #Get all tupples of elements in each list.
        #All elements 'under' this tupple are going to be a set in the partition i.e. they collide.
        Tuples = list(iterate_lists(lists))
        #Element 0 would be (0,...,0), which would mean, that we wouldn't pick any element.
        del Tuples[0]


        for element in Tuples:
            #left_marked is going to store the left over marked points that are not in the currently made partition set
            left_marked = list(marked_points)
            #part stores the marked points in the currently made partition set
            part = []
            for i in range(len(left_marked)):
                #Picking the left over marked points, lieing 'above' the others
                left_marked[i] = left_marked[i][element[i]:]
                part = part + list(marked_points[i][:element[i]])

            flatten_part = sorted(flatten(part), key = cmp_to_key(compare))


            if left_marked != [[]for i in range(len(marked_points))]:
                #Calculate the possible partitions of the left over marked points
                #'Delete' the 1-bracket above which there are no left marked points
                left_marked = [br for br in left_marked if br != []]
                res = Wn.get_partitions(left_marked)
            else:
                res = [[]]
            for item in res:
                #concatenate the results from before and add them to the list of final Partitions
                final_part.append([flatten_part] + item)
            
            
        
        return final_part
               
    def make_2br_over_1br(br, node):
        '''
        Input:  PosetNode node, which is a face of Wn.
                A 1-bracket br of the node.
        Output: List of all children of node, that come from a collision over br.
        '''
        if len(list(br)) == 1 and type(br[0]) == list:
            br = br[0]
        '''-----------FIND ALL IMPORTANT 2-Brackets---------------'''
        #1.2-Brackets over input 1-bracket br
        try: 
            #If there are 2-brackets above br set cur_2br to be these 2-brackets.
            cur_2br = node.d[str(flatten(br))]
            cur_2br = [[elt] for elt in cur_2br]
        except KeyError:
            cur_2br = []
        
        #2. Greatest 2-Brackets of 1-brackets in in(br)
        #We need to take a look at the greatest 2-brackets of the 1-brackets which are subsets of br, so that we don't violate (2-BRACKETING)
        #We make a bracket around the intergers so that we can access the belonging places in the dictionary.
        #bra = [[i] if isinstance(i, int) else i for i in br]
        bra = [i if isinstance(i, list) else [i] for i in br]
        #collect all 2-brackets as described
        smaller_2br = []
        for elt in bra:
            try:
                if node.d[str(flatten(elt))] != [] and node.d[str(flatten(elt))] != [[]]:
                    smaller_2br = smaller_2br + [(node.d[str(flatten(elt))])[0]]
            except KeyError:
                continue

        #3. Smallest 2-Brackets of next bigger bracket
        #Consider the smallest 2-brackets of the 1-bracket which is the smallest one in which br is contained so that (2-BRACKETING) isn't violated.
        #find_smallest method finds the described 1-bracket.
        biggerbr = find_smallest(node.a, br)
        try: 
            if biggerbr != None:
                bigger_2br = [node.d[str(flatten(biggerbr))][-1]]
            else: 
                bigger_2br = []
        except KeyError:
            bigger_2br = []

        #Having brackets around all the exsisting 2-brackets, since they have to be seperated by the underlying 1-bracket they are on
        arr_2br = [bigger_2br] + cur_2br + [smaller_2br]
        while [] in arr_2br:
            arr_2br.remove([])
        while [[]] in arr_2br:
            arr_2br.remove([[]])
        #Technical thing such that we don't have no repatition in the loop later. 
        #Only way this can happen is that there are no cur_2br points
        if bigger_2br == [] and arr_2br != []:
            #If biggerbr == [] the next bigger bracket is [0,....,r-1]. We need to add this seperatly because [0,...,r-1] is not in the dict from the beginning
            if len(arr_2br[0]) == 1 or cur_2br == []:
                arr_2br = [[[flatten(arr_2br[0])]]] + arr_2br
            else: 
                arr_2br = [[[flatten(arr_2br[0][0])]]] + arr_2br
        

        '''-------------------MAKE PARTITIONS------------------'''

        #Even if there are no 2-Brackets we want to make a collision of the seams.
        if cur_2br == [] and smaller_2br == [] and len(flatten(br)) > 1:
            partitions = [[]]
        else:
            #Just for storage
            br_partitions = [[[] for j in range(len(arr_2br[i][0]))] for i in range(len(arr_2br) - 1)]
            
            #Iterate threw the list of 2-brackets. For each 2-bracket find the smaller 2-brackets, that can collide to form new 2-brackets.
            for i in range(len(arr_2br) - 1):
                for j in range(len(arr_2br[i][0])):
                    #Get elements of smaller 2-bracket, that one can use to make collisions
                    av_elt = get_elt_of_smallerbr(arr_2br[i][0][j], arr_2br[i+1])
                    
                    br_partitions[i][j] = Wn.get_partitions(av_elt)
                    #We can't have collisions in the whole bracket due to dimenison reasons
                    if i > 0 and br_partitions[i][j] != [] and len(br_partitions[i][j]) > 1:
                        del br_partitions[i][j][-1]

            #In the following we distinguish between Type-2 and Type-1&-3 moves.
            #Type-1&-3 moves
            if cur_2br != [] and cur_2br != [[]]:
                partitions = []
                #Type-1 move is only the first entry of br_partitions TODO There must be a better way to do type 1 moves.
                if br_partitions != []:
                    for i in range(len(br_partitions)-1):
                        #For i = 0,...,len(br_partitions)-2 we have type-1 moves, since br is already in the 1-bracketing
                        while [] in br_partitions[i]:
                            br_partitions[i].remove([])
                        if len(br_partitions[i]) > 1:
                            #Out of the smaller collision in the brackets form one bigger collision by taking tuples.
                            tuples = list(itertools.product(*br_partitions[i])) 
                            #Change the format of the partitions in tuples to the one we want to use.
                            res = [sum(part, []) for part in tuples]
                        else:
                            if br_partitions[i] != []:
                                res = br_partitions[i][0]
                            else: res = []
                        partitions += res

                #For Type 1 moves we are only allowed to make 1 addtional 2-bracket. 
                indices = []
                for i in range(len(partitions)):
                    #Count the 2-brackets that are not contained in the previous node.
                    counter = 0
                    cur_lists = get_iso_lists(partitions[i])
                    prev_lists = get_iso_lists(node.d[str(flatten(br))])
                    #We also need to consider the 2-brackets in smaller_2br
                    for sub in smaller_2br:
                        prev_lists = prev_lists+ sub
                    for elt in cur_lists:
                        if elt not in prev_lists: counter = counter + 1
                    #If there are more then 1 new 2-brackets this is not a valid partition of the previous 2-brackets.
                    if counter >= 2: 
                        indices.append(i)
                for index in sorted(indices, reverse=True):
                    del partitions[index]
                        

                #Type 3 moves: The last collisions that appear are Type 3 moves.
                if br_partitions != []:
                    for j in range(len(br_partitions[-1])):
                        if br_partitions[-1][j] != [] and br_partitions != [[]]:
                            for k in range(len(br_partitions[-1][j])):
                                #Replace the current bracket with the new made partition                
                                new_part = copy.deepcopy(cur_2br[-1])
                                new_part = new_part[0][:j] + copy.deepcopy(br_partitions[-1][j][k]) + new_part[0][j+1:]
                                partitions.append(new_part)
            
            else: #Type 2 moves
                #Get Partitons out of the br_partitions by taking all the tuples of each br_partitions[i]
                partitions = []
                for i in range(len(br_partitions)):
                    #[] in br_partitions[i] only comes from the biggerbr, containing marked nodes that are not in the current bracket
                    if i  == 0:
                        while [] in br_partitions[i]:
                            br_partitions[i].remove([])
                    if len(br_partitions[i]) > 1:
                        #Out of the smaller collision in the brackets form one bigger collision
                        tuples = list(itertools.product(*br_partitions[i]))
                        #Change the format of the partitions in tuples to the one we want to use.
                        res = [sum(part, []) for part in tuples]   
                    else:
                        if br_partitions[i] != []:
                            res = br_partitions[i][0]
                        else: res = []
                    partitions = partitions + res

        '''-------------------MAKE FINAL NODES------------------'''

        Nodes = []
        #Go over every calculated collision and create new poset nodes.
        for item in partitions:
            #set new values for the node with the collision more
            b = flatten(br)
            d = dict(node.d)
            #If the dictionary at b is not yet set we want to set it to [].
            try:
                d[str(b)]
            except KeyError:
                d[str(b)] = []

            #Only make this bracket if it is not already in there!
            if list(item) in d[str(b)]:
                continue
            #We can't have a bracket around everything due to dimension reasons.
            if b == list(np.arange(len(node.n))) and len(item) == 1:
                continue 

            #We don't want to make 'collisions' that don't add anything to the current brackets.
            if is_double(d[str(b)], item):
                continue

            #If br is not already a bracket in a make it a bracket. This is exactly the case if we have a type2 move.
            if not b in get_iso_lists(node.a) and list(b) != list(np.arange(len(node.n))) and not len(b) <= 1:
                a = make_1_bracket(node.a, b[0], b[-1])
                
            else:
                #If it is already in there just set it to the previous one.
                a = list(node.a)

            #Sort the entries in the dict to maintain the outest bracket at front and the smallest at the end.
            d[str(b)] = sorted(d[str(b)]+ [list(item)], key = len)
            #If a list in the dictionary does not add a 2-bracket to the whole 2-brackets, delete it. 
            #d[str(b)] = remove_double(d[str(b)])

            newnode = PosetNode(d, a, node.n, node.codim + 1)
            #Append the nodes list with the new generated node
            Nodes.append(newnode)
        return Nodes

    def get_all_brackets(li):
        #Takes a list as an input and gives back all the possible new 1-brackets that can occur in a collision.
        #Similar to Kr.make_all_brackets.
        res = []
        l = len(li)
        #Starts with the outer bracket and makes a new in there, if there are more then three elements in there.
        if l >= 2:
            poss = choose(np.arange(l), 2)
            newbr = [li[idx[0]:idx[1] + 1] for idx in poss]
            res = res + newbr
        
        #If there are lists contained in the list recursivly make all possible brackets in the list
        for i in range(l):
            if type(li[i]) == list:
                #Produce all the subbrackets subbr in the 1-bracket li[i]
                res =  res + Wn.get_all_brackets(li[i])     
        return res
    
    def make_all_brackets(node):
        #Finds all 1-brackets. For Type 1&3 moves they are given by the 1-brackets of node.
        #For Type 2 moves they are given by the collisions that can happen in the underlying Associahedra.
        brackets = Wn.get_all_brackets(node.a)
        #Add all the singeltons
        brackets += [[i] for i in range(len(node.n))]
        nodes = []
        for br in brackets:
            #Calculate all the new 2-Bracketings over br
            nodes = nodes + Wn.make_2br_over_1br(br, node)
        return nodes

    def build(self):
        #Calculate the dimension of the 2-Associahedra
        dim = len(self.n) - 3
        for elt in self.n: dim = dim + elt

        #Initialize storage for faces. Faces, again refered as nodes, of diffrent dimension are stored in different dictionaries.
        Nodes = [{} for i in range(dim + 1)]
        #Use the string representation of root as key for the dict
        Nodes[0][string_representation(self.root)] = self.root

        Q = Queue()
        Q.put(self.root)
        #Run over every node and create all children.
        while not Q.empty():
            curnode = Q.get()
            newcodim = curnode.codim + 1
            childs = list(Wn.make_all_brackets(curnode))
            #Wr.make_all_brackets2(curnode)

            for child in childs: 
                #Find index where one can put the new generated child

                if string_representation(child) in Nodes[newcodim]:
                    innode = Nodes[newcodim][string_representation(child)]
                    curnode.add_child([innode])
                else:
                    #If not add the child as a node and put it in the Queue
                    curnode.add_child([child])
                    Nodes[newcodim][string_representation(child)] = child
                    Q.put(child)
        return Nodes
    
    def f_vector(self):
        #Prints out the f-vector of the poset.
        return list(reversed([len(self.nodes[i]) for i in range(len(self.nodes))]))
                    
    def alternating(self):
        #Prints out the alternating some of the f-vector. It has to be 1, since the 2-Associahedra is an abstract polytope.
        f_vector = self.f_vector()
        result = 0
        for i in range(len(f_vector)):
            if i % 2 == 0:
                result += f_vector[i]
            else:
                result -= f_vector[i]
        return result
                    


if __name__ == '__main__':
    
     
    W  = Wn([2,0,1])
    print(f'f_vector: {W.f_vector()}, alternating sum: {W.alternating()}, sum: {sum(W.f_vector())}')

    start = time.time()
    TwoAss_4_dimensional = generate_arrays(4)
    for n in TwoAss_4_dimensional:
        W = Wn(n)
        print(f'n: {n}, f_vector: {W.f_vector()}, alternating sum: {W.alternating()}, sum: {sum(W.f_vector())}')
    end = time.time()
    print('Time taken:',end - start,'seconds.')

    W = Wn([2,0,1])
    for i in range(len(W.nodes)):
        print(f'Codimension {i} faces:')
        for key in W.nodes[i]:
            print(W.nodes[i][str(key)].d)




