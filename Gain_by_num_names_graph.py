import os, math, json, random, csv
import numpy as np
from collections import defaultdict
from scipy import stats
from data_reader import *
from information_calculations import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm

"""
This script creates the panel of 3d plots showing information gain as a function of the round and the number of additional names the player sees. For each participant and each round, the gain is simulated by sampling the current number of names 200 times.
"""



def analyze_game(file_name, memory_length, divergence_type="JS"):
    """
    The primary method of the plotting. It calls the "simulate_information_gains" function in the
    the "information_calculations" module.  That function returns a list of divergences, one for each round. The value is the average of 200 divergence differences based on synthetic distributions of names.
    """

    # extracting the right data from the Otree CSV
    game_data = import_file(file_name)

    # packing the game data in useful ways
    participant_data, group_round_names = preanalysis_packing(game_data)

    # Applying "limited memory" to the whole system to discount names no longer in circulation
    true_distro_by_round = impose_limited_memory(group_round_names, memory_length)

    file_stem = current_file_name.split("-")


    # We store all the participants divergence averages here.
    three_d = [[[] for i in range(23)] for j in range(25)]

    for participant, data in participant_data.items():
        # getting the no-unstructured info distribution, the baseline for comparison.
        distro_by_round_no_unstructured = impose_limited_memory(data["distro_by_round_no_unstructured"],
                                                    memory_length=memory_length)

        for rnd in range(0,25):
            # sending the baseline off for simulated gains.
            divergence_avgs = simulate_information_gains(distro_by_round_no_unstructured[rnd+1],
                                                         true_distro_by_round[rnd+1],
                                                         divergence_type=divergence_type)
            # for each round, we record the participant's average for each number of names.
            for indx, divergence in enumerate(divergence_avgs):
                    three_d[rnd][indx].append(divergence)

    return three_d


def flatten_matrix(matrix):
    """
    Just packing up the numpy array the way pyplot.trisurf method wants.
    """
    x = []
    y = []
    z = []
    for i in range(25):
        for j in range(23):
            x.append(i)
            y.append(j)
            z.append(matrix[i][j])

    return x,y,z


def average_vals(three_d_group):
    """
    The third dimension of the array contains all the participants' average for the given round and number of names. This function averages those values. The array of averages is then flattened into three lists of 3d coordinates and returned.
    """
    new = np.zeros((25,23))
    for rnd in range(25):
        for add_name in range(23):
            vals = three_d_group[rnd][add_name]
            new[rnd,add_name] = sum(vals)/len(vals)
    return new


def merge_3d_lists(group, new):
    """
    Each game produces a new array of values. This function appends the new values to an array for recording multiple runs.
    """
    for rnd in range(0,25):
        for add_names in range(0,23):
            group[rnd][add_names].extend(new[rnd][add_names])



memory_length = 8
div_type = "KL" #JS or KL

# We combine all the individual run data into a lists for each of the six subplots.
random_success = [[[] for i in range(23)] for j in range(25)]
random_failure = [[[] for i in range(23)] for j in range(25)]
sw_success = [[[] for i in range(23)] for j in range(25)]
sw_failure = [[[] for i in range(23)] for j in range(25)]
lattice_success = [[[] for i in range(23)] for j in range(25)]
lattice_failure = [[[] for i in range(23)] for j in range(25)]
random_zero = [[[] for i in range(23)] for j in range(25)]
sw_zero = [[[] for i in range(23)] for j in range(25)]
lattice_zero = [[[] for i in range(23)] for j in range(25)]


for directory, sub, files in os.walk("../experiment_data"):
    for current_file_name in files:
        # We don't look at the files with no
        if "Data" in current_file_name and not "ignore" in current_file_name and not "MLM" in current_file_name:

            file_name = "/".join([directory,current_file_name])

            results = analyze_game(file_name, memory_length,divergence_type=div_type)

            # These runs resulted in a convention. The list is used to sort the data into successful and failed runs.
            convention_names = ["2addtl-RANDOMA","2addtl-RANDOMB","2addtl-RANDOMC",
                                    "2addtl-RANDOMD","2addtl-SMALLA","2addtl-SMALLB",
                                    "2addtl-SMALLC","2addtl-SMALLD","2addtl-LATTICED",
                                   "1addtl-RANDOMB","1addtl-RANDOMD","1addtl-SMALLC"]


            # creating the master  array of round by names values,
            run_name = current_file_name.replace("Data-","").replace(".csv","")
            if run_name in convention_names:
                if "RANDOM" in run_name:
                    merge_3d_lists(random_success,results)
                elif "LATTICE" in run_name:
                    merge_3d_lists(lattice_success,results)
                else:
                    merge_3d_lists(sw_success,results)
            else:
                if "RANDOM" in run_name:
                    if "0addtl-" in run_name:
                        merge_3d_lists(random_zero,results)
                    else:
                        merge_3d_lists(random_failure,results)
                elif "LATTICE" in run_name:
                    if "0addtl-" in run_name:
                        merge_3d_lists(lattice_zero,results)
                    else:
                        merge_3d_lists(lattice_failure,results)
                else:
                    if "0addtl-" in run_name:
                        merge_3d_lists(sw_zero,results)
                    else:
                        merge_3d_lists(sw_failure,results)

# creating the figure
fig = plt.figure(figsize=(10,16))

# random networks, successful runs, (upper left)
ax = fig.add_subplot(331, projection='3d')
ax.set_title("A: Random, successful")
ax.set_xlabel("Round No.")
ax.set_ylabel("Add'l Names")
ax.set_zlim(0,3)
ax.view_init(22, 35)
x,y,z = flatten_matrix(average_vals(random_success))
ax.plot_trisurf(x, y, z, cmap=plt.cm.viridis, linewidth=0.2)

#Random networks, failed runs (upper middle)
ax = fig.add_subplot(332, projection='3d')
ax.set_title("D: Random, failure")
ax.set_xlabel("Round No.")
ax.set_ylabel("Add'l Names")
ax.set_zlabel("Divergence Diff.")
ax.set_zlim(0,3)
ax.view_init(22, 35)
x,y,z = flatten_matrix(average_vals(random_failure))
ax.plot_trisurf(x, y, z, cmap=plt.cm.viridis, linewidth=0.2)

#Random networks, failed runs (upper right)
ax = fig.add_subplot(333, projection='3d')
ax.set_title("G: Random, no add'l names")
ax.set_xlabel("Round No.")
ax.set_ylabel("Add'l Names")
ax.set_zlabel("Divergence Diff.")
ax.set_zlim(0,3)
ax.view_init(22, 35)
x,y,z = flatten_matrix(average_vals(random_zero))
ax.plot_trisurf(x, y, z, cmap=plt.cm.viridis, linewidth=0.2)

# Small World, successful runs (middle left)
ax = fig.add_subplot(334, projection='3d')
ax.set_title("B: Small World, successful")
ax.set_xlabel("Round No.")
ax.set_ylabel("Add'l Names")
ax.set_zlim(0,3)
ax.view_init(22, 35)
x,y,z =flatten_matrix(average_vals(sw_success))
ax.plot_trisurf(x, y, z, cmap=plt.cm.viridis, linewidth=0.2)

# Small Wordl, failed runs (middle middle)
ax = fig.add_subplot(335, projection='3d')
ax.set_title("E: Small World, failure")
ax.set_zlabel("Divergence Diff.")
ax.set_xlabel("Round No.")
ax.set_ylabel("Add'l Names")
ax.set_zlim(0,3)
ax.view_init(22, 35)
x,y,z = flatten_matrix(average_vals(sw_failure))
ax.plot_trisurf(x, y, z, cmap=plt.cm.viridis, linewidth=0.2)

# Small Wordl, failed runs (middle right)
ax = fig.add_subplot(336, projection='3d')
ax.set_title("H: Small World, no add'l names")
ax.set_zlabel("Divergence Diff.")
ax.set_xlabel("Round No.")
ax.set_ylabel("Add'l Names")
ax.set_zlim(0,3)
ax.view_init(22, 35)
x,y,z = flatten_matrix(average_vals(sw_zero))
ax.plot_trisurf(x, y, z, cmap=plt.cm.viridis, linewidth=0.2)

# lattice networks, successful runs (bottom left)
ax = fig.add_subplot(337, projection='3d')
ax.set_title("C: Lattice, successful")
ax.set_xlabel("Round No.")
ax.set_ylabel("Add'l Names")
ax.set_zlim(0,3)
ax.view_init(22, 35)
x,y,z = flatten_matrix(average_vals(lattice_success))
ax.plot_trisurf(x, y, z, cmap=plt.cm.viridis, linewidth=0.2)

# lattice networks, failed runs (bottom middle)
ax = fig.add_subplot(338, projection='3d')
ax.set_title("F: Lattice, failure")
ax.set_zlabel("Divergence Diff.")
ax.set_xlabel("Round No.")
ax.set_ylabel("Add'l Names")
ax.set_zlim(0,3)
ax.view_init(22, 35)
x,y,z = flatten_matrix(average_vals(lattice_failure))
ax.plot_trisurf(x, y, z, cmap=plt.cm.viridis, linewidth=0.2)

# lattice networks, failed runs (bottom right)
ax = fig.add_subplot(339, projection='3d')
ax.set_title("I: Lattice, no add'l names")
ax.set_zlabel("Divergence Diff.")
ax.set_xlabel("Round No.")
ax.set_ylabel("Add'l Names")
ax.set_zlim(0,3)
ax.view_init(22, 35)
x,y,z = flatten_matrix(average_vals(lattice_zero))
ax.plot_trisurf(x, y, z, cmap=plt.cm.viridis, linewidth=0.2)

# Just tightening the layout up a bit
plt.subplots_adjust(left=.125, bottom=0.15, right=.9, top=0.7, wspace=.1, hspace=.2)

plt.savefig("../analysis/Gain_by_names_{}_mem_{}_zeros.png".format(div_type,memory_length),dpi=300)
