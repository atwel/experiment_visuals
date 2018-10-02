import math
import json
import random
import csv
from collections import defaultdict
import os
import matplotlib.pyplot as plt

"""
This script can read the data files from the individual trials of the
experiment. Those data files are just CSVs with a lot of
information related to the management of the trials and cheating checks.
To make everything more manageable, there is file with the list of the
important variables names. Unneeded variables and records--because of the
particulars of the Mechanical Turk platform, many experiments have more
than required number of participants and those participants played in
isolation--are removed during the import process, i.e. import_file()

The second function takes the imported data and packages it to be more
accessible for analysis.
"""



def import_file(file_name):

    # load of the list of variable names we actually want
    with open("fields_to_keep_no_survey.csv","r") as f:
        raw = csv.reader(f)
        keepers = next(raw)


    massive_dic = {}
    print("Importing {}".format(file_name))
    # standard load of csv data file.
    with open(file_name,"r") as f:
        data = csv.reader(f,dialect="excel")
        header = next(data)
        turk_id_index = header.index("participant.mturk_worker_id")
        participant_number = header.index("instructions.1.player.index_trans")

        for i in data:
            # The third column in the data is an indicator of whether
            # the participant spot was active during the trial
            if i[2] != "Inactive":
                massive_dic[int(i[participant_number])] = i

        indices = []
        # Going through the header to find the indices of the variable
        # nams we actually want to keep
        for i in keepers:
            indices.append(header.index(i))

        # Now we just loop over the participants we need, grab the
        # variables we need and pack them into a dictionary.
        cleaned_data = {}
        for i,j in massive_dic.items():
            holder = {}
            for k in keepers:
                try:
                    holder[k] = j[header.index(k)]
                except:
                    pass
            cleaned_data[i] = holder


        """
        At this point, there are still active participants in the
        data dictionary that we don't need (they are overflowed
        participants who played by themselves). We need to identify
        they and remove they. (We needed their data to properly
        identify them. Specifically their group id needs to be less
        than 12.
        """
        IDs_to_delete = []
        for ID,info in cleaned_data.items():
            group_id = info["coordinate.1.group.id_in_subsession"]

            if int(group_id) > 12:
                IDs_to_delete.append(ID)
                #print("deleted",group_id)
        # We don't delete immediately because Python doesn't like dictionaries
        # being touched during any iterative process.
        for ID in IDs_to_delete:
            del cleaned_data[ID]


    return cleaned_data


def import_file_for_MLM(file_name):

    participant_dictionary = {}
    print("Importing {}".format(file_name))
    # standard load of csv data file.
    with open(file_name,"r") as f:
        data = csv.reader(f,dialect="excel")
        header = next(data)
        turk_id_index = header.index("participant.mturk_worker_id")
        participant_number = header.index("instructions.1.player.index_trans")

        for i in data:
            # The third column in the data is an indicator of whether
            # the participant spot was active during the trial
            if i[2] != "Inactive":
                participant_dictionary[int(i[participant_number])] = i

            # load of the list of variable names we actually want
        with open("fields_to_keep_MLM.csv","r") as f:
            raw = csv.reader(f)
            keepers = next(raw)

        column_names = ["network_type","network_version","additional_names_count"]
        for column_name in keepers:
            if "unstructured" in column_name:
                new_name = column_name.replace("unstructured", "random_name_")

                column_names.append(new_name+"1")
                column_names.append(new_name+"2")
            else:
                column_names.append(column_name)


        # Now we just loop over the participants we need, grab the
        # variables we need and pack them into a dictionary.
        new_file_name = file_name.replace(".csv","_MLM.csv")

        run_data = file_name.split("/")[-1].replace(".csv","").split("-")
        print(run_data)
        network_version = run_data[-1][-1]
        network_type = run_data[-1][:-1]
        additional_names_count = run_data[1][0]

        with open(new_file_name,"w") as f:
            f.write(",".join(column_names)+"\n")
            for i,j in participant_dictionary.items():
                if int(j[header.index("coordinate.1.group.id_in_subsession")]) <= 12:
                    holder = [network_type, network_version,additional_names_count]
                    for k in keepers:
                        if "unstructured" in k:
                            names = j[header.index(k)].split(",")

                            holder.extend(names)
                            if len(names) == 1:
                                holder.append("")

                        else:
                            holder.append(j[header.index(k)])
                    f.write(",".join(holder)+"\n")



"""
We need to repack the data in a way that makes more sense for the
questions we ask. The preanalysis_packing function does that.
"""
def preanalysis_packing(game_data):

    participant_data = {}
    group_round_names = defaultdict(list)

    # the game data is currently organized as a dictionary with
    # (participant, values) pairs. So the outer loop is for
    # individual participants
    for key, values in game_data.items():
        participant_names_played = []
        alter_names_played = []
        group_numbers = []
        unstructured_names = []
        distro_by_round_no_unstructured = defaultdict(list)
        distro_by_round_unstructured = defaultdict(list)


        # Now we loop over the rounds of game
        for i in range(1,26):

            # The name the participant played that round
            name_played = values["coordinate.{}.player.display_name".format(i)].lower().strip()
            # adding to the list of all names played
            group_round_names[i].append(name_played)
            # adding to the participant's own list
            participant_names_played.append(name_played)

            # The name the participant's alter played
            alter_name_played = values["coordinate.{}.player.alter".format(i)].lower().strip()
            alter_names_played.append(alter_name_played)

            group_numbers.append(values["coordinate.{}.group.id_in_subsession".format(i)].lower().strip())

            # The additional "niche" names the participant was explored to.
            unstructured_string = values["coordinate.{}.player.stigmergy".format(i)].lower()

            # repacking
            all_round_names = [name_played,alter_name_played]
            distro_by_round_no_unstructured[i] = list(all_round_names)

            # The niche names were recorded as a single string and are parsed here.
            if unstructured_string != "":
                stig = [j.strip().lower() for j in unstructured_string.split(',')]
                unstructured_names.append(stig)
                all_round_names.extend(stig)

            distro_by_round_unstructured[i] = all_round_names


        participant_data[key] = {"names_played":participant_names_played,
                                  "alters_played":alter_names_played,
                                  "unstructured_names": unstructured_names,
                                  "distro_by_round_unstructured":distro_by_round_unstructured,
                                  "distro_by_round_no_unstructured":distro_by_round_no_unstructured,
                                  "group_numbers":group_numbers
                                  }


    for key, values in participant_data.items():
        actual_alters = {}
        for round, group in enumerate(values["group_numbers"]):
                for alter in range(24):
                    if  alter != key and participant_data[alter]["group_numbers"][round] == group:
                        actual_alters[round+1] = alter
                        break
        assert len(actual_alters) == 25
        values["actual_alters"] = actual_alters

    return (participant_data, group_round_names)
