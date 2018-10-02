import os,math, json, random, csv
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from collections import defaultdict
from scipy import stats
from data_reader import *
from information_calculations import *

"""
This script creates the panel of plots with the differences in divergences. There are three comparisons appearing in three columns and 3 network types, but each with two outcomes (successful, failed), so there are 6 rows. Each plot has the group average difference with the standard deviation bounds.
"""

def analyze_game(file_name, memory_length, divergence_type="JS"):
    # extracting the right data from the Otree CSV
    game_data = import_file(file_name)

    # packing the game data in useful ways
    participant_data, group_round_names = preanalysis_packing(game_data)

    # Applying "limited memory" to the whole system to discount names no longer in circulation
    true_distro_by_round = impose_limited_memory(group_round_names, memory_length)


    file_stem = current_file_name.split("-")
    network_topology_name = file_stem[-1].replace(".csv",".txt")
    additional_names_count = int(file_stem[-2].replace("addtl",""))

    # The first comparision type
    most_info_results = find_non_neighbors_with_most_info(participant_data,
                                                            true_distro_by_round,
                                                            network_topology_name,
                                                            additional_names_count,
                                                            divergence_type=divergence_type,
                                                            memory_length=memory_length)

    # Second comparison
    random_others_results = find_random_others(participant_data,true_distro_by_round,
                                                network_topology_name,
                                                additional_names_count,
                                                only_non_neighbors=False,
                                                divergence_type=divergence_type,
                                                memory_length=memory_length)

    # third comparison
    weak_ties_results = simulate_weakest_ties(participant_data,true_distro_by_round,
                                                network_topology_name,
                                                additional_names_count,
                                                divergence_type=divergence_type,
                                                memory_length=memory_length)

    # fourth comparison
    spare_results = find_spare_alter(participant_data,true_distro_by_round,
                                        network_topology_name,
                                        additional_names_count,
                                        only_non_neighbors=False,
                                        divergence_type=divergence_type,
                                        memory_length=memory_length)

    return (most_info_results,random_others_results, weak_ties_results, spare_results)


# packing up the list of participants values into three lists of the mean and the standard dev.
def get_run_averages(the_list):
    avg = []
    upper_se = []
    lower_se = []
    for rnd in the_list:
        rnd_avg = np.mean(rnd)
        se = np.std(rnd)
        avg.append(rnd_avg)
        upper_se.append(rnd_avg+se)
        lower_se.append(rnd_avg-se)

    return (avg, upper_se, lower_se)

# Finding the max ratio score for each round
def get_max_ratio(the_list):
    maxs = []
    for rnd in the_list:
        maxs.append(max(rnd))
    return maxs

# Function to make and dress up the plots
def make_subplot(averages, ratios, row, column, label):
    axarr[row,column].spines["top"].set_visible(False)
    axarr[row,column].spines["bottom"].set_visible(False)
    axarr[row,column].grid(True,linestyle='-', linewidth=.25)
    axarr[row,column].set_xlim([1, 25])
    axarr[row,column].text(2, .75,label, fontsize=16,fontweight="bold")
    if div_type == "KL":
        axarr[row,column].set_ylim([-.5,1])
    else:
        axarr[row,column].set_ylim([-.075,.15])
    axarr[row,column].axhline(0,0,24,color="k",linewidth=.7)
    axarr[row,column].tick_params(axis='y',color="gray",direction="inout")
    axarr[row,column].tick_params(axis='x',length=0)
    axarr[row,column].spines['left'].set_color('lightgray')
    axarr[row,column].spines['right'].set_color('lightgray')
    if row !=5:
        axarr[row,column].set_xticklabels([])

    x_range = range(1,26)
    axarr[row, column].plot(x_range, averages[0],"b", linewidth=1.75)
    axarr[row, column].plot(x_range, averages[1], color="gray", linestyle="dashed", linewidth=.75)
    axarr[row, column].plot(x_range, averages[2], color="gray", linestyle="dashed", linewidth=.75)

    right_ax = axarr[row, column].twinx()
    right_ax.plot(x_range, ratios, "r", linewidth=1.25)
    right_ax.set_ylim([-.5,1])
    right_ax.spines["top"].set_visible(False)
    right_ax.spines["bottom"].set_visible(False)
    right_ax.tick_params(axis='y',color="gray",direction="inout")
    right_ax.tick_params(axis='x',length=0)
    right_ax.spines['right'].set_color('lightgray')
    right_ax.spines['left'].set_color('lightgray')
    if column == 3:
        right_ax.set_ylabel("Divergence Ratio",fontdict={"fontsize":14},labelpad=15, rotation=270)


# The giant block below is where we pack all the data up  the way we need to

memory_length = 8
div_type = "KL" #JS or KL


sw_alter_success = [[] for i in range(25)]
sw_random_success = [[] for i in range(25)]
sw_weak_success = [[] for i in range(25)]
sw_most_success = [[] for i in range(25)]
sw_alter_failure = [[] for i in range(25)]
sw_random_failure = [[] for i in range(25)]
sw_weak_failure = [[] for i in range(25)]
sw_most_failure = [[] for i in range(25)]

sw_alter_suc_ratio = [[] for i in range(25)]
sw_random_suc_ratio = [[] for i in range(25)]
sw_weak_suc_ratio= [[] for i in range(25)]
sw_most_suc_ratio = [[] for i in range(25)]
sw_alter_fail_ratio = [[] for i in range(25)]
sw_random_fail_ratio = [[] for i in range(25)]
sw_weak_fail_ratio = [[] for i in range(25)]
sw_most_fail_ratio = [[] for i in range(25)]

random_alter_success = [[] for i in range(25)]
random_random_success = [[] for i in range(25)]
random_weak_success = [[] for i in range(25)]
random_most_success = [[] for i in range(25)]
random_alter_failure = [[] for i in range(25)]
random_random_failure = [[] for i in range(25)]
random_weak_failure = [[] for i in range(25)]
random_most_failure = [[] for i in range(25)]

random_alter_suc_ratio = [[] for i in range(25)]
random_random_suc_ratio = [[] for i in range(25)]
random_weak_suc_ratio = [[] for i in range(25)]
random_most_suc_ratio = [[] for i in range(25)]
random_alter_fail_ratio = [[] for i in range(25)]
random_random_fail_ratio = [[] for i in range(25)]
random_weak_fail_ratio= [[] for i in range(25)]
random_most_fail_ratio = [[] for i in range(25)]


lat_alter_success = [[] for i in range(25)]
lat_random_success = [[] for i in range(25)]
lat_weak_success = [[] for i in range(25)]
lat_most_success = [[] for i in range(25)]
lat_alter_failure = [[] for i in range(25)]
lat_random_failure = [[] for i in range(25)]
lat_weak_failure = [[] for i in range(25)]
lat_most_failure = [[] for i in range(25)]

lat_alter_suc_ratio = [[] for i in range(25)]
lat_random_suc_ratio = [[] for i in range(25)]
lat_weak_suc_ratio = [[] for i in range(25)]
lat_most_suc_ratio = [[] for i in range(25)]
lat_alter_fail_ratio = [[] for i in range(25)]
lat_random_fail_ratio = [[] for i in range(25)]
lat_weak_fail_ratio = [[] for i in range(25)]
lat_most_fail_ratio = [[] for i in range(25)]



for directory, sub, files in os.walk("../experiment_data"):
    for current_file_name in files:
        if "Data" in current_file_name and not "ignore" in current_file_name and not "0addtl-" in current_file_name and not "MLM" in current_file_name:

            file_name = "/".join([directory,current_file_name])

            results = analyze_game(file_name, memory_length,divergence_type=div_type)

            convention_names = ["2addtl-RANDOMA","2addtl-RANDOMB","2addtl-RANDOMC",
                                "2addtl-RANDOMD","2addtl-SMALLA","2addtl-SMALLB",
                                "2addtl-SMALLC","2addtl-SMALLD","2addtl-LATTICED",
                               "1addtl-RANDOMB","1addtl-RANDOMD","1addtl-SMALLC"]


            run_name = current_file_name.replace("Data-","").replace(".csv","")
            if run_name in convention_names:
                if "RANDOM" in run_name:
                    for participant, values in results[0].items():
                        diffs = values[0]
                        ratio = values[3]
                        for rnd in range(25):
                            random_most_success[rnd].append(diffs[rnd])
                            random_most_suc_ratio[rnd].append(ratio[rnd])
                    for participant, values in results[1].items():
                        diffs = values[0]
                        ratio = values[3]
                        for rnd in range(25):
                            random_random_success[rnd].append(diffs[rnd])
                            random_random_suc_ratio[rnd].append(ratio[rnd])
                    for participant, values in results[2].items():
                        diffs = values[0]
                        ratio = values[3]
                        for rnd in range(25):
                            random_weak_success[rnd].append(diffs[rnd])
                            random_weak_suc_ratio[rnd].append(ratio[rnd])
                    for participant, values in results[3].items():
                        diffs = values[0]
                        ratio = values[3]
                        for rnd in range(25):
                            random_alter_success[rnd].append(diffs[rnd])
                            random_alter_suc_ratio[rnd].append(ratio[rnd])

                elif "LATTICE" in run_name:
                    for participant, values in results[0].items():
                        diffs = values[0]
                        ratio = values[3]
                        for rnd in range(25):
                            lat_most_success[rnd].append(diffs[rnd])
                            lat_most_suc_ratio[rnd].append(ratio[rnd])
                    for participant, values in results[1].items():
                        diffs = values[0]
                        ratio = values[3]
                        for rnd in range(25):
                            lat_random_success[rnd].append(diffs[rnd])
                            lat_random_suc_ratio[rnd].append(ratio[rnd])
                    for participant, values in results[2].items():
                        diffs = values[0]
                        ratio = values[3]
                        for rnd in range(25):
                            lat_weak_success[rnd].append(diffs[rnd])
                            lat_weak_suc_ratio[rnd].append(ratio[rnd])
                    for participant, values in results[3].items():
                        diffs = values[0]
                        ratio = values[3]
                        for rnd in range(25):
                            lat_alter_success[rnd].append(diffs[rnd])
                            lat_alter_suc_ratio[rnd].append(ratio[rnd])

                else:
                    for participant, values in results[0].items():
                        diffs = values[0]
                        ratio = values[3]
                        for rnd in range(25):
                            sw_most_success[rnd].append(diffs[rnd])
                            sw_most_suc_ratio[rnd].append(ratio[rnd])
                    for participant, values in results[1].items():
                        diffs = values[0]
                        ratio = values[3]
                        for rnd in range(25):
                            sw_random_success[rnd].append(diffs[rnd])
                            sw_random_suc_ratio[rnd].append(ratio[rnd])
                    for participant, values in results[2].items():
                        diffs = values[0]
                        ratio = values[3]
                        for rnd in range(25):
                            sw_weak_success[rnd].append(diffs[rnd])
                            sw_weak_suc_ratio[rnd].append(ratio[rnd])
                    for participant, values in results[3].items():
                        diffs = values[0]
                        ratio = values[3]
                        for rnd in range(25):
                            sw_alter_success[rnd].append(diffs[rnd])
                            sw_alter_suc_ratio[rnd].append(ratio[rnd])
            else:
                if "RANDOM" in run_name:
                    for participant, values in results[0].items():
                        diffs = values[0]
                        ratio = values[3]
                        for rnd in range(25):
                            random_most_failure[rnd].append(diffs[rnd])
                            random_most_fail_ratio[rnd].append(ratio[rnd])
                    for participant, values in results[1].items():
                        diffs = values[0]
                        ratio = values[3]
                        for rnd in range(25):
                            random_random_failure[rnd].append(diffs[rnd])
                            random_random_fail_ratio[rnd].append(ratio[rnd])
                    for participant, values in results[2].items():
                        diffs = values[0]
                        ratio = values[3]
                        for rnd in range(25):
                            random_weak_failure[rnd].append(diffs[rnd])
                            random_weak_fail_ratio[rnd].append(ratio[rnd])
                    for participant, values in results[3].items():
                        diffs = values[0]
                        ratio = values[3]
                        for rnd in range(25):
                            random_alter_failure[rnd].append(diffs[rnd])
                            random_alter_fail_ratio[rnd].append(ratio[rnd])

                elif "LATTICE" in run_name:
                    for participant, values in results[0].items():
                        diffs = values[0]
                        ratio = values[3]
                        for rnd in range(25):
                            lat_most_failure[rnd].append(diffs[rnd])
                            lat_most_fail_ratio[rnd].append(ratio[rnd])
                    for participant, values in results[1].items():
                        diffs = values[0]
                        ratio = values[3]
                        for rnd in range(25):
                            lat_random_failure[rnd].append(diffs[rnd])
                            lat_random_fail_ratio[rnd].append(ratio[rnd])
                    for participant, values in results[2].items():
                        diffs = values[0]
                        ratio = values[3]
                        for rnd in range(25):
                            lat_weak_failure[rnd].append(diffs[rnd])
                            lat_weak_fail_ratio[rnd].append(ratio[rnd])
                    for participant, values in results[3].items():
                        diffs = values[0]
                        ratio = values[3]
                        for rnd in range(25):
                            lat_alter_failure[rnd].append(diffs[rnd])
                            lat_alter_fail_ratio[rnd].append(ratio[rnd])

                else:
                    for participant, values in results[0].items():
                        diffs = values[0]
                        ratio = values[3]
                        for rnd in range(25):
                            sw_most_failure[rnd].append(diffs[rnd])
                            sw_most_fail_ratio[rnd].append(ratio[rnd])
                    for participant, values in results[1].items():
                        diffs = values[0]
                        ratio = values[3]
                        for rnd in range(25):
                            sw_random_failure[rnd].append(diffs[rnd])
                            sw_random_fail_ratio[rnd].append(ratio[rnd])
                    for participant, values in results[2].items():
                        diffs = values[0]
                        ratio = values[3]
                        for rnd in range(25):
                            sw_weak_failure[rnd].append(diffs[rnd])
                            sw_weak_fail_ratio[rnd].append(ratio[rnd])
                    for participant, values in results[3].items():
                        diffs = values[0]
                        ratio = values[3]
                        for rnd in range(25):
                            sw_alter_failure[rnd].append(diffs[rnd])
                            sw_alter_fail_ratio[rnd].append(ratio[rnd])



# Now we actually make the figure and push everything into it
f, axarr = plt.subplots(6,4,figsize=(28, 20))

tt1 = axarr[0, 0].set_title('Random Neighbors',fontdict={"fontsize":20})
tt1.set_position([.5, 1.10])
tt2 = axarr[0, 1].set_title('Random Others',fontdict={"fontsize":20})
tt2.set_position([.5, 1.10])
tt3 = axarr[0, 2].set_title('Weak Ties',fontdict={"fontsize":20})
tt3.set_position([.5, 1.10])
tt4 = axarr[0, 3].set_title('Max Info. Others',fontdict={"fontsize":20})
tt4.set_position([.5, 1.10])

for row in range(0,5,2):
    axarr[row, 0].set_ylabel("Divergence\nDifference",fontdict={"fontsize":14},labelpad=7)

for row in range(1,6,2):
    axarr[row, 0].set_ylabel("Divergence\nDifference",fontdict={"fontsize":14},labelpad=7)

for column in range(4):
    axarr[5, column].set_xlabel("Round number",fontdict={"fontsize":16},labelpad=18)
    plt.setp([a.get_xticklabels() for a in axarr[5, :]], fontsize=16)

f.text(s="Random Networks", x=0.04, y=.88,fontdict={"fontsize":24},rotation=90)
f.text(s="Small World Networks", x=0.04, y=.68,fontdict={"fontsize":24},rotation=90)
f.text(s="Lattice Networks", x=0.04, y=.45,fontdict={"fontsize":24},rotation=90)

f.text(s="Successful Runs", x=0.07, y=.9,fontdict={"fontsize":16,"color":"green"},rotation=90)
f.text(s="Failed Runs", x=0.07, y=.78,fontdict={"fontsize":16,"color":"red"},rotation=90)
f.text(s="Successful Runs", x=0.07, y=.69,fontdict={"fontsize":16,"color":"green"},rotation=90)
f.text(s="Failed Runs", x=0.07, y=.57,fontdict={"fontsize":16,"color":"red"},rotation=90)
f.text(s="Successful Run", x=0.07, y=.47,fontdict={"fontsize":16,"color":"green"},rotation=90)
f.text(s="Failed Runs", x=0.07, y=.36,fontdict={"fontsize":16,"color":"red"},rotation=90)

plt.subplots_adjust(bottom=0.3, right=.8, top=0.9, hspace=.6)



# random alter success
make_subplot(get_run_averages(random_alter_success),get_max_ratio(random_alter_suc_ratio),0,0,"A")

# random alter failure
make_subplot(get_run_averages(random_alter_failure),get_max_ratio(random_alter_fail_ratio),1,0, "B")

#random random success
make_subplot(get_run_averages(random_random_success),get_max_ratio(random_random_suc_ratio),0,1,"G")

#random random failure
make_subplot(get_run_averages(random_random_failure),get_max_ratio(random_random_fail_ratio),1,1,"H")

#random max_info success
make_subplot(get_run_averages(random_weak_success),get_max_ratio(random_weak_suc_ratio),0,2,"M")

#random max_info failure
make_subplot(get_run_averages(random_weak_failure),get_max_ratio(random_weak_fail_ratio),1,2, "N")

#random max_info success
make_subplot(get_run_averages(random_most_success),get_max_ratio(random_most_suc_ratio),0,3, "S")

#random max_info failure
make_subplot(get_run_averages(random_most_failure),get_max_ratio(random_most_fail_ratio),1,3, "T")



# sw alter success
make_subplot(get_run_averages(sw_alter_success),get_max_ratio(sw_alter_suc_ratio),2,0,"C")

# sw alter failure
make_subplot(get_run_averages(sw_alter_failure),get_max_ratio(sw_alter_fail_ratio),3,0,"D")

# sw random success
make_subplot(get_run_averages(sw_random_success),get_max_ratio(sw_random_suc_ratio),2,1,"I")

# sw random failure
make_subplot(get_run_averages(sw_random_failure),get_max_ratio(sw_random_fail_ratio),3,1,"J")

# sw most success
make_subplot(get_run_averages(sw_weak_success),get_max_ratio(sw_weak_suc_ratio),2,2,"O")

# sw most failure
make_subplot(get_run_averages(sw_weak_failure),get_max_ratio(sw_weak_fail_ratio),3,2,"P")

# sw most success
make_subplot(get_run_averages(sw_most_success),get_max_ratio(sw_most_suc_ratio),2,3,"U")

# sw most failure
make_subplot(get_run_averages(sw_most_failure),get_max_ratio(sw_most_fail_ratio),3,3,"V")




# lat alter success
make_subplot(get_run_averages(lat_alter_success),get_max_ratio(lat_alter_suc_ratio),4,0,"E")

# lat alter failure
make_subplot(get_run_averages(lat_alter_failure),get_max_ratio(lat_alter_fail_ratio),5,0,"F")

# lat random success
make_subplot(get_run_averages(lat_random_success),get_max_ratio(lat_random_suc_ratio),4,1,"K")

# lat random failure
make_subplot(get_run_averages(lat_random_failure),get_max_ratio(lat_random_fail_ratio),5,1,"L")

# lat most success
make_subplot(get_run_averages(lat_weak_success),get_max_ratio(lat_weak_suc_ratio),4,2,"Q")

# lat most failure
make_subplot(get_run_averages(lat_weak_failure),get_max_ratio(lat_weak_fail_ratio),5,2,"R")

# lat most success
make_subplot(get_run_averages(lat_most_success),get_max_ratio(lat_most_suc_ratio),4,3,"W")

# lat most failure
make_subplot(get_run_averages(lat_most_failure),get_max_ratio(lat_most_fail_ratio),5,3,"X")




plt.savefig("../analysis/Divergence_diffs_{}_mem_{}.png".format(div_type,memory_length),dpi=300,bbox_inches='tight')
