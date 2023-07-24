from holes import Holes 
import random
import numpy as np
from helperfunctions import *

cpu_cores = 2

shape = [100, 100, 100]
samples = 1
n_cubes = np.empty((25, shape[0]//2, samples))
p_vec = np.linspace(0.0, 0.5, 50)
rounds_max = 30

def reset_seed(p, seed, shape):
    """
    Randomly measure qubits.
    """
    D = Holes(shape)
    removed_nodes = np.zeros(shape[0]*shape[1]*shape[2], dtype=bool)

    random.seed(int(seed))
    # p is the probability of losing a qubit

    measurementChoice = 'Z'
    for i in range(shape[0]*shape[1]*shape[2]):
        if random.random() < p:
            removed_nodes[i] = True
            D.add_node(i)
        if i % 10000000 == 0:
            print(i/(shape[0]*shape[1]*shape[2])*100)
    return D, removed_nodes


def algorithm1(D, removed_nodes, shape):
    holes = D.graph.nodes
    hole_locations = np.zeros(8)

    #counting where the holes are
    for h in holes:
        x, y, z = h
        for zoffset in range(2):
            for yoffset in range(2):
                for xoffset in range(2):
                    if ((x + xoffset) % 2 == (z + zoffset) % 2) and ((y + yoffset) % 2 == (z + zoffset) % 2):
                        hole_locations[xoffset+yoffset*2+zoffset*4] += 1
    
    xoffset = np.argmax(hole_locations) % 2
    yoffset = np.argmax(hole_locations) // 2
    zoffset = np.argmax(hole_locations) // 4

    for z in range(shape[2]):
        for y in range(shape[1]):
            for x in range(shape[0]):
                if ((x + xoffset) % 2 == (z + zoffset) % 2) and ((y + yoffset) % 2 == (z + zoffset) % 2):
                    i = get_node_index(x, y, z, shape)
                    removed_nodes[i] = True
    
    return xoffset, yoffset, zoffset

import pickle

def main(input):
    """
    input = list containing [probability, seed] 
    """
    start = time.time()

    p, seed, rounds = input
    
    data = np.zeros(samples)

    D, removed_nodes = reset_seed(p, seed, shape)
    print('done building grid', f'p = {p}, samples={seed}/{samples}')

    if rounds > 0:
        for i in range(rounds):
            repairs, failures = D.repair_grid(p)

        removed_nodes = np.zeros(shape[0]*shape[1]*shape[2], dtype=bool)
        for f in failures:
            i = get_node_index(*f, shape)
            removed_nodes[i] = True

    xoffset, yoffset, zoffset = algorithm1(D, removed_nodes, shape)
    cubes, n_cubes = D.findlattice(removed_nodes, xoffset, yoffset, zoffset)
    print('latticies found', f'p = {p}, samples={seed}/{samples}')

    


    C = D.build_centers_graph(cubes)
    
    #with open(f'./data/ncubes{p:.4f}shape{shape[0]}sample{seed}', 'wb') as f:
    #    pickle.dump(n_cubes, f)

    #connected_cubes = D.findconnectedlattice(C)      
    #with open(f'./data/cubes{p:.4f}shape{shape[0]}sample{seed}', 'wb') as f:
    #    pickle.dump(connected_cubes, f)
    
    largestcc = D.findmaxconnectedlattice(C)
    with open(f'./data/cc{p:.4f}shape{shape[0]}{shape[1]}{shape[2]}sample{seed}round{rounds}', 'wb') as f:
        pickle.dump(largestcc, f)

    end1loop = time.time()
    print((end1loop-start)/60, 'mins elapsed', f'p = {p}, samples={seed}/{samples}')
    return 



import matplotlib.pyplot as plt
import time
import multiprocessing as mp



input_vec = [(p, s, r) for p in p_vec for s in range(samples) for r in range(0, rounds_max, 5)]

if __name__ == "__main__":
    start = time.time()
    print(input_vec)
    pool = mp.Pool(processes=cpu_cores)
    results = pool.map(main, input_vec)
    pool.close()
    pool.join()

    #n_cubes = np.vstack(results)
    connected_cubes_len = np.array([results])
        
    print((time.time() - start)/60)
    """
    np.save('data_connected_cubes.npy', connected_cubes_len)
    print(connected_cubes_len.shape, p_vec.shape)

    plt.figure()
    plt.scatter(p_vec, connected_cubes_len, label = f'shape = {shape}, cubesize={1}')
    plt.xlabel('p')
    plt.title('Number of connected subgraphs vs. p')
    plt.ylabel('N')
    plt.legend()

    plt.savefig(f'connectedsubgraph{shape[0]}.png')
    """

