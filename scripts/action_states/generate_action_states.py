import numpy as np
import itertools
import os
import csv

from numpy.lib.function_base import diff

num_nodes = 9


object_to_index = {}

path_prefix = os.path.dirname(__file__)

#read the object and bin information from the corresponding csv files. This is user input.
def read_objects_and_bins(path):
    bins = {}
    objects = []

    with open(path + 'objects.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    for x in data:
        objects.append(x)
    
    with open(path + 'bins.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    
    #set what each colored bin corresponds to which node in the graph
    for x in data:
        bins[x[0]] = int(x[1])

    return objects, bins

#Generate all possible states
def generate_states(objects):
    ls_nodes = list(range(num_nodes))
    ls_complete = [ls_nodes]

    #iterate through all objects and append their states to the cartesian product
    for i, obj in enumerate(objects):
        object_to_index[str(obj)] = i+1
        ls_complete.append([0,1])
    
    #find possible states from cartesian product
    temp = itertools.product(*ls_complete)
    states = list([list(tup) for tup in temp])
    return states

#generate the state-action matrix 
def generate_action_matrix(states, objects, bins):
    n = len(states)
    action_matrix = np.ones((n,n), dtype=int) * (-1)
    
    #for each state look at all actions and the next states from those actions
    for i, state in enumerate(states):
        for obj in objects:
            index = object_to_index[str(obj)]
            color = obj[1]
            #if the object hasn't been moved then move it and calculate next state
            if state[index] == 0:
                nextstate = state.copy()
                nextstate[index] = 1
                nextstate[0] = bins[color]
                state_index = states.index(nextstate)  
                if i != state_index:
                    action_matrix[i, state_index] = index -1 
    return action_matrix

#save the states and state-action matrix to csv
def save_data(data, name):
    """ Saves a list to a csv file using numpy """

    np.savetxt(name, data, fmt='%5s', delimiter = ',')




if __name__=="__main__":
    objects, bins = read_objects_and_bins(path_prefix)

    states = generate_states(objects)
    save_data(states, 'states.csv')

    action_matrix = generate_action_matrix(states, objects, bins)
    save_data(action_matrix, "action_matrix.csv")