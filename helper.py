import itertools
from functools import cmp_to_key
import ast

#Help functions. Not all of them are used in the construction of the algorithm, but might be helpful for other purposes.

def flatten(nested_list):
    #Flattens a nested list
    flattened_list = []
    for item in nested_list:
        if type(item) == list:
            flattened_list += flatten(item)
        else:
            flattened_list.append(item)
    return flattened_list

def get_iso_lists(l):
    '''
    Input: Nested list
    Output: All Isolated lists flattend
    '''
    iso_lists = []
        
    for item in l:
        if type(item) == list:
            iso_lists.append(flatten(item))
            res = get_iso_lists(item)
            iso_lists = iso_lists + res
    return iso_lists

def iterate_lists(lists):
    #Iterates threw a list
    for elements in itertools.product(*lists):
        yield elements

def get_elt_of_smallerbr(br, collision):
    #Given a 2-bracket br in arr_2br, the function returns all of the 2-brackets in the collision list which are subset of br, sorted by strata.
    res = []
    for stratum in collision: 
        l = list([])
        for item in stratum:
            if set(item).issubset(set(br)):
                l.append(item)
        res = res + [l]
    return res

def find_smallest(nested_list, target_list):
    #Input: nested_list: 1-bracketing of some face of the 2-associahedra
    #       target_list: 1-bracket inside the 1-bracketing
    #Output: Finds the next bigger 1-bracket in the 1-bracketing.
    if len(list(target_list)) == 1 and type(target_list[0]) == list:
        target_list = target_list[0]
    bool_val = True
    for elt in target_list:
        try:
            if not elt in nested_list:
                bool_val = False
        except ValueError:
            bool_val = False
    for l in nested_list:
        if str(target_list) == str(l): 
            bool_val = True
    if bool_val and str(nested_list) != str(target_list):
        return nested_list
    for l in nested_list:
        if type(l) == list:
            res = find_smallest(l, target_list)
            if res != None:
                return res
    return None

def compare(a,b):
    #given two 2-tuples of integers, it compares them by first comparing the first element and then the second
    if a[0] != b[0]:
        if a[0] < b[0]:
            return -1
        else:
            return 1
    else:
        if a[1] < b[1]:
            return -1
        elif a[1] > b[1]:
            return 1
        else: return 0 


def make_1_bracket(lst, i, j):
    #Makes new 1-bracket in lst around elements i and j.
    lststr = str(lst)
    idx = lststr.find(str(i))
    lststr = lststr[:idx] + '[' + lststr[idx:]
    idx = lststr.find(str(j))
    lststr = lststr[:idx+1] + ']' + lststr[idx+1:]

    lst = ast.literal_eval(lststr)

    return lst



def is_double(lt, col):
    #Input: lt: list of collisions out of the dictionary of a face.
    #       col: One collision in the dict.
    #Output: boolean that indicates if the collision adds any new 2-bracket to lt.
    iso_set_lt = set(tuple(inner_list) for inner_list in get_iso_lists(lt))
    iso_set_col = set(tuple(inner_list) for inner_list in get_iso_lists(col))
    if iso_set_col.issubset(iso_set_lt) and len(col) != 1 and col != []:
        return True
    else:
        return False
    
def remove_double(lt):
    #Removes the collisions in lt, that don't add any new 2-bracket to the rest.
    bad_i = []
    for i in range(len(lt)):
        iso_set_lt = set(tuple(inner_list) for inner_list in get_iso_lists(lt[:i]+ lt[i+1:]))
        iso_set_br = set(tuple(inner_list) for inner_list in get_iso_lists(lt[i]))
        if iso_set_br.issubset(iso_set_lt):
            bad_i.append(i)
        bad_i.sort(reverse= True)
    for index in bad_i:
        if len(lt[index]) > 1:
            del lt[index]
    return lt

def choose_iter(elements, length):
        for i in range(len(elements)):
            if length == 1:
                yield (elements[i],)
            else:
                for next in choose_iter(elements[i+1:], length-1):
                    yield (elements[i],) + next

def choose(l, k):
    #Calles the function choose_iter to return all the possibilities to pick k elements out of l as a list.
    return list(choose_iter(l, k))

def compare_tuples(tup1, tup2):
    #Compares list of pairs of integers.
    if len(tup1) < len(tup2):
        return -1
    elif len(tup1) > len(tup2):
        return 1
    else:
        for i in range(len(tup1)):
            if tup1[i][0] < tup2[i][0]:
                return -1
            elif tup1[i][0] > tup2[i][0]:
                return 1
            else:
                if tup1[i][1] < tup2[i][1]:
                    return -1
                elif tup1[i][1] > tup2[i][1]:
                    return 1
    return 0

def get_isolated_brackets(brackets):
    #Takes as input the dictionary inputs of one 1-bracket and spits out all the 2-brackets that are over this 1-bracket. (Function \beta).
    iso_list = []
    for i in range(len(brackets)):
        iso_list = iso_list + get_iso_lists(brackets[i])
    indices = []
    for i in range(len(iso_list)):
        if iso_list[i] in iso_list[i+1:]:
            indices.append(i)
    for i in sorted(indices, reverse = True):
        del iso_list[i]

    iso_set = set(tuple(inner_list) for inner_list in iso_list)
    return sorted(iso_set, key = cmp_to_key(compare_tuples))

def compare_orders(node1, node2):
    #Comparison function on faces of the 2-associahedra that is not dependent on the way of storage.
    #Works similar to the string representation.
    if str(node1.a) < str(node2.a):
        return True
    elif str(node1.a) > str(node2.a):
        return False
    else: 
        #The only possibility that they do not have the same keys is that one dictionary containes the key [0,1,...,r]
        for key in sorted(node2.d):
            try:
                node1.d[str(key)]
            except KeyError:
                return False
        for key in sorted(node1.d):
            try:
                node2.d[str(key)]
            except KeyError:
                return True
            
            iso_bracket1 = get_isolated_brackets(node1.d[str(key)])
            iso_bracket2 = get_isolated_brackets(node2.d[str(key)])

            if str(iso_bracket1) < str(iso_bracket2):
                return True
            elif str(iso_bracket1) > str(iso_bracket2):
                return False
            else:
                if str(flatten(node1.d[str(key)][-1])) < str(flatten(node2.d[str(key)][-1])):
                    return True
                elif str(flatten(node1.d[str(key)][-1])) > str(flatten(node2.d[str(key)][-1])):
                    return False
                

        return None

    
    

def string_representation(f):
    '''
    This function takes a node and returns the representation of it as a string to access the dict
    '''
    f_str = str(f.a)
    for key in sorted(f.d):
        f_str += str(key)
        f_str += str(get_isolated_brackets(f.d[str(key)]))
        f_str += str(f.d[str(key)][-1])
    return f_str


def generate_arrays(dim):
    #Generates all arrays that belong to 2-associahedra of dimension dim, except for permutations.
    dim = dim + 3
    arrays = []
    for length in range(1, dim):
        for combination in itertools.combinations_with_replacement(range(dim - length + 1), length):
            if (len(list(combination)) + sum(list(combination))) == dim:
                arrays.append(list(combination)) 
    return arrays





