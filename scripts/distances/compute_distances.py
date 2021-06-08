import numpy as np

'''
returns the min distances between each node and the shortest path for each pair of nodes
n is the number of nodes
weight_mat is the adjacency mat of the graph
'''
def floyd_warshall(n, weight_mat):
    distance = np.zeros((n,n), dtype = int)
    Next = np.zeros((n,n), dtype = int)

    #initialize shortest distance and path helper to decide what node is next in the path
    #Next[i,j] denotes the next node to take in the path from i to j
    for i in range(n):
        for j in range(n):
            distance[i,j] = weight_mat[i,j]

            if (weight_mat[i,j] > 1e7):
                Next[i,j] = -1
            else:
                Next[i,j] = j

    #compute shortest paths using dynamic programming
    for k in range(n):
        for i in range(n):
            for j in range(n):
                
                if i == j:
                    continue
                #if going through the intermediate node is shorter, update the shortest path 
                if distance[i,k] + distance[k,j] < distance[i,j]:
                    distance[i,j] = distance[i,k] + distance[k,j]
                    Next[i,j] = Next[i,k]

    #find shortest path from Next by iterating on it
    #i is the first node, Next[i,j] is the second, Next[Next[i,j],j] is the third etc.
    def find_shortest_path(i,j):
        if Next[i,j] == -1:
            return []
        
        shortest_path = [i]
        while i != j:
            i = Next[i,j]
            shortest_path.append(i)
        return shortest_path

    #write down all shortest paths for each pair
    paths = [[None for x in range(n)] for y in range(n)]
    for i in range(n):
        for j in range(n):
            path = find_shortest_path(i,j)
            paths[i][j] = path

    return distance, paths



if __name__=="__main__":
    weight_mat = np.genfromtxt('./map1_matrix.csv', delimiter=',')

    n = weight_mat.shape[0]

    #set 0 dist to infinity for the algorithm 
    for i in range(weight_mat.shape[0]):
        for j in range(weight_mat.shape[1]):
            if weight_mat[i,j] == 0:
                weight_mat[i,j] = 1e9

    distance, paths = floyd_warshall(n, weight_mat)

    #write the shortest paths to a file
    with open("shortest_paths.txt", "w") as file:
        file.write(str(paths))

    #also save a readable version for debugging purposes
    with open("readable_shortest_paths.txt", "w") as file:
        for i in range(n):
            for j in range(n):
                file.write('The path from {} to {} is: '.format(i,j) + str(paths[i][j]) + '\n' )

    print(distance)
    print(paths)
    np.save('distances.npy', distance)