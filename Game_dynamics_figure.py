import math
import json
import random
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from data_reader import *


def GetEntropy(L,max):
    """ Return the entropy of a list of probabilities."""

    entropy = 0.0
    while len(L) < max:
        L.append(0)
    total_count = len(L)
    percentages = [i/total_count for i in L]

    max_entropy = -math.log(1.0/total_count)

    for p in percentages:
        if not p == 0:
            entropy -= p*math.log(p)

    try:
        return entropy / max_entropy
    except:
        return 0



def GetHHI(L,max):
    """
    Return the normalized Herfindahl-Hirschmann Index of the list L.
    """
    while len(L) < max:
        L.append(0)
    total_count = len(L)
    total = sum(L)
    percentages = [i/total for i in L]

    HHI_unnormed = sum([i**2 for i in percentages])

    HHI_normed = (HHI_unnormed - 1/total_count)/(1 - 1/total_count)

    return HHI_normed

def smooth_data(line):
    new_line = []
    new_line.append(sum(line[:2])/2.)
    for i in range(1,24):

        new_line.append(sum(line[i-1:i+2])/3.)
    new_line.append(sum(line[-2:])/2.)

    return new_line

def compile_data(game_data):

    #calculating the number of names alive
    name_life = {}
    name_start = {}
    round_none_counts = defaultdict(int)
    for values in game_data.values():
        for rnd in range(1,26):
            name = values["coordinate.{}.player.display_name".format(rnd)].lower()
            if name == "(none given)" or name =="(none)":
                name = "(none)"
                round_none_counts[rnd] += 1
            else:
                if name not in name_life.keys():
                    name_start[name] = rnd
                    name_life[name] = rnd
                elif rnd > name_life[name]:
                    name_life[name] = rnd
                try:
                    if name_start[name] > rnd:
                        name_start[name] = rnd
                except:
                    pass

    count_plot = []
    for rnd in range(1,26):
        cnt = 0
        for name,val in name_life.items():
            if rnd >= name_start[name] and rnd <= val:
                cnt +=1
        if rnd in round_none_counts.keys():
            cnt += round_none_counts[rnd]

        count_plot.append(cnt)

    # calculating the success rate
    successes = defaultdict(int)
    for values in game_data.values():
        for i in range(1,26):
            match = int(values["coordinate.{}.player.success".format(i)].lower())
            if match:
                successes[i] += 1
    for i in range(1,26):
        if i not in successes.keys():
            successes[i]=0

    success_rate = []
    for i in range(1,26):
        success_rate.append(successes[i]/24.)
        # We've counted each individual's success and so have double counted the each interaction.
        # Dividing by 24 halves each, so the final value is the % of all 24 interactions.

    # calculating the entropy of the names
    name_histos = defaultdict(lambda:defaultdict(int))
    for values in game_data.values():
        for i in range(1,26):
            name = values["coordinate.{}.player.display_name".format(i)].lower()
            if name == "(none given)":
                name = "(none)"
            name_histos[i][name] +=1

    entropies = []

    for i in range(1,26):
        vals=[]
        if "(none)" in name_histos[i].keys():
            singles = name_histos[i]["(none)"]
            vals = [1 for j in range(singles)]
        for key, val in name_histos[i].items():
            if key != "(none)":
                vals.append(val)
        entropies.append(GetEntropy(vals,24))

    HHI = []

    for i in range(1,26):
        vals=[]
        if "(none)" in name_histos[i].keys():
            singles = name_histos[i]["(none)"]
            vals = [1 for j in range(singles)]
        for key, val in name_histos[i].items():
            if key != "(none)":
                vals.append(val)
        HHI.append(GetHHI(vals,24))

    return (count_plot,smooth_data(success_rate), smooth_data(entropies), smooth_data(HHI))



# ZERO names Full

game_data = import_file("../experiment_data/Data-0addtl-FULLB.csv") # import_data is from the data_reader module
fullA_counts, fullA_success, fullA_entropy, fullA_HHI = compile_data(game_data)

game_data = import_file("../experiment_data/Data-0addtl-FULLC.csv")
fullB_counts, fullB_success, fullB_entropy, fullB_HHI = compile_data(game_data)


#ZERO names Random

game_data = import_file("../experiment_data/Data-0addtl-RANDOMC.csv")
random0A_counts, random0A_success, random0A_entropy, random0A_HHI = compile_data(game_data)

game_data = import_file("../experiment_data/Data-0addtl-RANDOMB.csv")
random0B_counts, random0B_success, random0B_entropy, random0B_HHI = compile_data(game_data)

#ZERO names Small World

game_data = import_file("../experiment_data/Data-0addtl-SMALLC.csv")
sw0A_counts, sw0A_success, sw0A_entropy, sw0A_HHI = compile_data(game_data)

game_data = import_file("../experiment_data/Data-0addtl-SMALLA.csv")
sw0B_counts, sw0B_success, sw0B_entropy, sw0B_HHI = compile_data(game_data)

#ZERO names Lattice

game_data = import_file("../experiment_data/Data-0addtl-LATTICEC.csv")
lattice0A_counts, lattice0A_success, lattice0A_entropy, lattice0A_HHI = compile_data(game_data)

game_data = import_file("../experiment_data/Data-0addtl-LATTICED.csv")
lattice0B_counts, lattice0B_success, lattice0B_entropy, lattice0B_HHI = compile_data(game_data)

# ONE name Random
game_data = import_file("../experiment_data/Data-1addtl-RANDOMA.csv")
random1F_counts, random1F_success, random1F_entropy, random1F_HHI = compile_data(game_data)

game_data = import_file("../experiment_data/Data-1addtl-RANDOMD.csv")
random1S_counts, random1S_success, random1S_entropy, random1S_HHI = compile_data(game_data)

# ONE name Small World

game_data = import_file("../experiment_data/Data-1addtl-SMALLA.csv")
sw1F_counts, sw1F_success, sw1F_entropy, sw1F_HHI = compile_data(game_data)

game_data = import_file("../experiment_data/Data-1addtl-SMALLC.csv")
sw1S_counts, sw1S_success, sw1S_entropy, sw1S_HHI = compile_data(game_data)


# ONE name Lattice
game_data = import_file("../experiment_data/Data-1addtl-LATTICEA.csv")
lattice1_counts, lattice1_success, lattice1_entropy, lattice1_HHI = compile_data(game_data)

# TWO name Random

game_data = import_file("../experiment_data/Data-2addtl-RANDOMA.csv")
random2A_counts, random2A_success, random2A_entropy, random2A_HHI = compile_data(game_data)

game_data = import_file("../experiment_data/Data-2addtl-RANDOMB.csv")
random2B_counts, random2B_success, random2B_entropy, random2B_HHI = compile_data(game_data)


# TWO name Small World

game_data = import_file("../experiment_data/Data-2addtl-SMALLA.csv")
sw2A_counts, sw2A_success, sw2A_entropy, sw2A_HHI = compile_data(game_data)

game_data = import_file("../experiment_data/Data-2addtl-SMALLD.csv")
sw2B_counts, sw2B_success, sw2B_entropy, sw2B_HHI = compile_data(game_data)

# TWO name Lattice

game_data = import_file("../experiment_data/Data-2addtl-LATTICEB.csv")
lattice2F_counts, lattice2F_success, lattice2F_entropy, lattice2F_HHI = compile_data(game_data)

game_data = import_file("../experiment_data/Data-2addtl-LATTICED.csv")
lattice2S_counts, lattice2S_success, lattice2S_entropy, lattice2S_HHI = compile_data(game_data)


# Now that we've imported and processed the data, we can plot

f, axarr = plt.subplots(3, 3,figsize=(20, 15))

tt1 = axarr[0, 0].set_title('Name Count',fontdict={"fontsize":24})
tt1.set_position([.5, 1.10])
tt2 = axarr[0, 1].set_title('Concentration (HHI)',fontdict={"fontsize":24})
tt2.set_position([.5, 1.10])
tt3 = axarr[0, 2].set_title('Success Rate',fontdict={"fontsize":24})
tt3.set_position([.5, 1.10])

axarr[0, 0].set_ylabel("Zero Add'l Names",fontdict={"fontsize":24},labelpad=25)
axarr[1, 0].set_ylabel("One Add'l Name",fontdict={"fontsize":24},labelpad=25)
axarr[2, 0].set_ylabel("Two Add'l Names",fontdict={"fontsize":24},labelpad=25)

axarr[0, 2].set_ylabel("%",fontdict={"fontsize":20},
                                    labelpad=-40,rotation="horizontal",verticalalignment="top")
axarr[1, 2].set_ylabel("%",fontdict={"fontsize":20},
                                    labelpad=-40,rotation="horizontal",verticalalignment="top")
axarr[2, 2].set_ylabel("%",fontdict={"fontsize":20},
                                    labelpad=-40,rotation="horizontal",verticalalignment="top")

axarr[2, 0].set_xlabel("Round number",fontdict={"fontsize":20},labelpad=22)
axarr[2, 1].set_xlabel("Round number",fontdict={"fontsize":20},labelpad=22)
axarr[2, 2].set_xlabel("Round number",fontdict={"fontsize":20},labelpad=22)

full_line = mlines.Line2D([], [], color='g',linewidth=4)
random_line = mlines.Line2D([], [], color='b',linewidth=4)
random_line_fail = mlines.Line2D([], [], color='b',linestyle="dashed",linewidth=4)
sw_line = mlines.Line2D([], [], color='r',linewidth=4)
sw_line_fail = mlines.Line2D([], [], color='r',linestyle="dashed",linewidth=4)
lattice_line = mlines.Line2D([], [], color='k',linewidth=4)
lattice_line_fail = mlines.Line2D([], [], color='k',linestyle="dashed",linewidth=4)

x = range(1,26)

## 0 name treatment

# Counts
axarr[0, 0].plot(x, fullA_counts, "g", linewidth=3)
axarr[0, 0].plot(x, fullB_counts, "g", linewidth=3)
axarr[0, 0].plot(x, random0A_counts, "b--", linewidth=3)
axarr[0, 0].plot(x, random0B_counts, "b--", linewidth=3)
axarr[0, 0].plot(x, sw0A_counts, "r--", linewidth=3)
axarr[0, 0].plot(x, sw0B_counts, "r--", linewidth=3)
axarr[0, 0].plot(x, lattice0A_counts, "k--", linewidth=3)
axarr[0, 0].plot(x, lattice0B_counts, "k--", linewidth=3)
axarr[0, 0].text(2, 20, "A", fontsize=48, fontweight="black")

# Standardized HHI diversity
axarr[0, 1].plot(x,fullA_HHI,"g",linewidth=3)
axarr[0, 1].plot(x,fullA_HHI,"g",linewidth=3)
axarr[0, 1].plot(x,random0A_HHI,"b--",linewidth=3)
axarr[0, 1].plot(x,random0A_HHI,"b--",linewidth=3)
axarr[0, 1].plot(x,sw0A_HHI,"r--",linewidth=3)
axarr[0, 1].plot(x,sw0A_HHI,"r--",linewidth=3)
axarr[0, 1].plot(x,lattice0A_HHI,"k--",linewidth=3)
axarr[0, 1].plot(x,lattice0A_HHI,"k--",linewidth=3)
axarr[0, 1].text(2,.8 , "D", fontsize=48, fontweight="black")

# Success Rate
axarr[0, 2].plot(x, fullA_success, "g", linewidth=3)
axarr[0, 2].plot(x, fullB_success, "g", linewidth=3)
axarr[0, 2].plot(x, random0A_success, "b--", linewidth=3)
axarr[0, 2].plot(x, random0B_success, "b--", linewidth=3)
axarr[0, 2].plot(x, sw0A_success, "r--", linewidth=3)
axarr[0, 2].plot(x, sw0B_success, "r--", linewidth=3)
axarr[0, 2].plot(x, lattice0A_success, "k--", linewidth=3)
axarr[0, 2].plot(x, lattice0B_success, "k--", linewidth=3)
axarr[0, 2].text(2, .8, "G", fontsize=48, fontweight="black")




## 1 name treatment

# Counts
axarr[1, 0].plot(x, random1F_counts, "b--", linewidth=3)
axarr[1, 0].plot(x, sw1F_counts, "r--", linewidth=3)
axarr[1, 0].plot(x, lattice1_counts, "k--", linewidth=3)
axarr[1, 0].plot(x, random1S_counts, "b", linewidth=3)
axarr[1, 0].plot(x, sw1S_counts, "r", linewidth=3)
axarr[1, 0].text(2, 20, "B", fontsize=48, fontweight="black")

# Standardized HHI diversity
axarr[1, 1].plot(x,random1F_HHI,"b--",linewidth=3)
axarr[1, 1].plot(x,sw1F_HHI,"r--",linewidth=3)
axarr[1, 1].plot(x,lattice1_HHI,"k--",linewidth=3)
axarr[1, 1].plot(x,random1S_HHI,"b",linewidth=3)
axarr[1, 1].plot(x,sw1S_HHI,"r",linewidth=3)
axarr[1, 1].text(2, .8, "E", fontsize=48, fontweight="black")

# Success Rate
axarr[1, 2].plot(x, random1F_success, "b--", linewidth=3)
axarr[1, 2].plot(x, sw1F_success, "r--", linewidth=3)
axarr[1, 2].plot(x, lattice1_success, "k--", linewidth=3)
axarr[1, 2].plot(x, random1S_success, "b", linewidth=3)
axarr[1, 2].plot(x, sw1S_success, "r", linewidth=3)
axarr[1, 2].text(2, .8, "H", fontsize=48, fontweight="black")




# 2 name treatment

# Counts
axarr[2, 0].plot(x, random2A_counts, "b", linewidth=3)
axarr[2, 0].plot(x, random2B_counts, "b", linewidth=3)
axarr[2, 0].plot(x, sw2A_counts, "r", linewidth=3)
axarr[2, 0].plot(x, sw2B_counts, "r", linewidth=3)
axarr[2, 0].plot(x, lattice2F_counts, "k--", linewidth=3)
axarr[2, 0].plot(x, lattice2S_counts, "k", linewidth=3)
axarr[2, 0].text(2, 20, "C", fontsize=48, fontweight="black")

# Standardize HHI diversity
axarr[2, 1].plot(x,random2A_HHI,"b",linewidth=3)
axarr[2, 1].plot(x,random2B_HHI,"b",linewidth=3)
axarr[2, 1].plot(x,sw2A_HHI,"r",linewidth=3)
axarr[2, 1].plot(x,sw2B_HHI,"r",linewidth=3)
axarr[2, 1].plot(x,lattice2F_HHI,"k--",linewidth=3)
axarr[2, 1].plot(x,lattice2S_HHI,"k",linewidth=3)
axarr[2, 1].text(2, .8, "F", fontsize=48, fontweight="black")

# Success Rate
axarr[2, 2].plot(x, random2A_success, "b", linewidth=3)
axarr[2, 2].plot(x, random2B_success, "b", linewidth=3)
axarr[2, 2].plot(x, sw2A_success, "r", linewidth=3)
axarr[2, 2].plot(x, sw2B_success, "r", linewidth=3)
axarr[2, 2].plot(x, lattice2F_success, "k--", linewidth=3)
axarr[2, 2].plot(x, lattice2S_success, "k", linewidth=3)
axarr[2, 2].text(2, .8, "I", fontsize=48, fontweight="black")





axarr[2, 0].legend([full_line, random_line,random_line_fail,sw_line,sw_line_fail,lattice_line,lattice_line_fail],
                   ("Full","Random","Random (failed)","Small","Small (failed)", "Lattice","Lattice (failed)"),
                   loc="upper right",fontsize=16,bbox_to_anchor=(1.1, 1.10),fancybox=True,shadow=True,)

axarr[0,0].grid(True)
axarr[0,1].grid(True)
axarr[0,2].grid(True)

axarr[1,0].grid(True)
axarr[1,1].grid(True)
axarr[1,2].grid(True)

axarr[2,0].grid(True)
axarr[2,1].grid(True)
axarr[2,2].grid(True)

axarr[0,1].set_ylim([0, 1])
axarr[0,2].set_ylim([0, 1])
axarr[0,0].set_xlim([1, 25])
axarr[0,1].set_xlim([1, 25])
axarr[0,2].set_xlim([1, 25])

axarr[0,1].set_ylim([0, 1])
axarr[0,2].set_ylim([0, 1])
axarr[1,0].set_xlim([1, 25])
axarr[1,1].set_xlim([1, 25])
axarr[1,2].set_xlim([1, 25])

axarr[2,1].set_ylim([0, 1])
axarr[2,2].set_ylim([0, 1])
axarr[2,0].set_xlim([1, 25])
axarr[2,1].set_xlim([1, 25])
axarr[2,2].set_xlim([1, 25])

plt.setp([a.get_xticklabels() for a in axarr[2, :]], fontsize=18)

plt.savefig("../analysis/Experiment_dynamics.png",dpi=300,bbox_inches='tight')
