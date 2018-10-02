import os,math, json, random, csv
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from collections import defaultdict
from scipy import stats
from data_reader import *
from information_calculations import *

"""
This script calculates the experiment wide averages (for both games where a convention emerged and games where they didn't) for the fraction of particpant matches where the name that was ultimately match on was first seen via the unstructured source and then only seen via the unstructured source. There are many of path ways through which the participant could be influenced by the exposure to the unstructured source, but unpacking those woul be quite hard.  For example, the participant might first see the name via the structured source, but then later twice via the unstructured source and then play it. The variants are numerous and finding the true regularities wouldn't necessarily enrich our knowledge, so I instead focus on the coarser counts above.
"""

def match_prior_exposures(participant_data, file_name):
    """
    This function takes the data of a single game and calculates the fraction of participant first-matches (i.e. the first time they matched on a name. Many participants will have multiple first matches, some one, some none) where the name matched upon first appeared via the unstructured source and then only via the unstructured source.
    """
    game_fraction_unstructured_before_structured = 0 #name seen via unstructured before structured
    game_fraction_only_unstructured = 0 # name seen only via unstructured

    total_matches = 0 # because participants can have more than 1 match, we need to ca
    for participant, values in participant_data.items():
        names = values["names_played"] # list by round of names participant played
        alters = values["alters_played"] # list by round of names interaction partners played
        unstructured = values["unstructured_names"] # list by round of names seen via unstructured

        # list comprehension calculation of rounds with matches and the matching name
        matches = [i.lower() if i.lower()==j.lower() and i != "(none)" else 0 for i,j in zip(names,alters) ]
        # the list looks something like [0,0,0, "Jon", "Jon", 0, 0, "Albert", "Albert", "Albert"]

        # now we need to go back and find the first matches only
        new_matches_only = []
        not_new = []
        for name in matches:
            if name != 0 and name not in not_new:
                new_matches_only.append(name)
                not_new.append(name) # conditional above will now be false for that name
            else:
                new_matches_only.append(0)
        # we now have a list like [0,0,0, "Jon", 0, 0, 0, "Albert", 0, 0]

        # we now go back to find when that name first appeared, if it did, for both sources
        match_prior_exposures = {}
        for indx, match in enumerate(new_matches_only):
            if match != 0:
                previously_seen_alters = []
                # we look over only the previous rounds, hence slice at indx
                for inner_indx, name in enumerate(alters[:indx]):
                    if name.lower() == match:
                        # found match in the names partner played and note the round
                        previously_seen_alters.append(inner_indx)

                previously_seen_unstructured = []
                # we look over only the previous rounds, hence slice at indx
                for inner_indx, names in enumerate(unstructured[:indx]):
                    # can get multiple names from the source so names is a list
                    cleaned_names = [nm.lower() for nm in names]
                    for nm in cleaned_names:
                        # can see the same name twice in the same round
                        if nm == match:
                            # found match with unstructured source name
                            previously_seen_unstructured.append(inner_indx)
                match_prior_exposures[indx] = (previously_seen_alters, previously_seen_unstructured)
        # For each match we now have a list of rounds before the match the same name was presented


        unstructured_before_structured = 0
        only_unstructured = 0
        count_matches = 0
        for structured_seen, unstructured_seen in match_prior_exposures.values():
            count_matches += 1
            if structured_seen == []: # didn't see via structured
                if unstructured_seen != []: # saw via unstructured
                    unstructured_before_structured += 1
                    only_unstructured += 1
                else: # didn't see anywhere. Likely playing same name and others matched to it.
                    pass
            else: # saw name via structured source
                if unstructured_seen != []: # also saw via unstructured source
                    if min(unstructured_seen) < min(structured_seen):
                        # if the lowest index of unstructured_seen is less than the lowest for structured_seen, as DJ Khaled says, another one!
                        unstructured_before_structured += 1

        # we now calculate the fraction of matches meeting the criteria for the individual and add it to the group/game tally
        if count_matches != 0:
            game_fraction_unstructured_before_structured +=                  unstructured_before_structured/count_matches
            game_fraction_only_unstructured += only_unstructured/count_matches
            #total_matches += count_matches

    cnt = len(participant_data)
    return (game_fraction_unstructured_before_structured/cnt, game_fraction_only_unstructured/cnt)


def analyze_game(file_name, memory_length, divergence_type="JS"):
    # extracting the right data from the Otree CSV
    game_data = import_file(file_name)

    # packing the game data in useful ways
    participant_data, group_round_names = preanalysis_packing(game_data)

    return match_prior_exposures(participant_data, file_name)



"""
We now initialize some parameters and counters and then loop over all the games while recording the results.
"""

memory_length = 8
div_type = "KL" #JS or KL

convention_names = ["2addtl-RANDOMA","2addtl-RANDOMB","2addtl-RANDOMC",
                    "2addtl-RANDOMD","2addtl-SMALLA","2addtl-SMALLB",
                    "2addtl-SMALLC","2addtl-SMALLD","2addtl-LATTICED",
                    "1addtl-RANDOMB","1addtl-RANDOMD","1addtl-SMALLC"]

successful_unstructured_before_structured_1 = 0
successful_only_unstructured_1 = 0
unsuccessful_unstructured_before_structured_1 = 0
unsuccessful_only_unstructured_1 = 0

successful_unstructured_before_structured_2 = 0
successful_only_unstructured_2 = 0
unsuccessful_unstructured_before_structured_2 = 0
unsuccessful_only_unstructured_2 = 0

successful_count_1 = 0
unsuccessful_count_1 = 0
successful_count_2 = 0
unsuccessful_count_2 = 0

for directory, sub, files in os.walk("../experiment_data"):
    for current_file_name in files:
        if "Data" in current_file_name and not "ignore" in current_file_name and not "0addtl" in current_file_name and not "MLM" in current_file_name:

            file_name = "/".join([directory,current_file_name])

            results = analyze_game(file_name, memory_length,divergence_type=div_type)
            print("Fraction unstructured b4 structured: {} \n Fraction only unstructured: {}".format(*results))


            run_name = current_file_name.replace("Data-","").replace(".csv","")
            if run_name in convention_names:
                if "1addtl" in run_name:
                        successful_unstructured_before_structured_1 += results[0]
                        successful_only_unstructured_1 += results[1]
                        successful_count_1 += 1
                else:
                        successful_unstructured_before_structured_2 += results[0]
                        successful_only_unstructured_2 += results[1]
                        successful_count_2 += 1

            else:
                if "1addtl" in run_name:
                    unsuccessful_unstructured_before_structured_1 += results[0]
                    unsuccessful_only_unstructured_1 += results[1]
                    unsuccessful_count_1+= 1
                elif "2addtl":
                    unsuccessful_unstructured_before_structured_2 += results[0]
                    unsuccessful_only_unstructured_2 += results[1]
                    unsuccessful_count_2+= 1
                else:
                    pass

print("Successful UBS1:", round(successful_unstructured_before_structured_1/successful_count_1,2))
print("Sucessful OU1:", round(successful_only_unstructured_1/successful_count_1,2))
print("Successful UBS2:", round(successful_unstructured_before_structured_2/successful_count_2, 2))
print("Sucessful OU2:", round(successful_only_unstructured_2/successful_count_2, 2))

print("Unsuccessful UBS:", round(unsuccessful_unstructured_before_structured_1/unsuccessful_count_1, 2))
print("Unsuccessful OU:",round(unsuccessful_only_unstructured_1/unsuccessful_count_1, 2))
print("Unsuccessful UBS:", round(unsuccessful_unstructured_before_structured_2/unsuccessful_count_2, 2))
print("Unsuccessful OU:", round(unsuccessful_only_unstructured_2/unsuccessful_count_2, 2))
