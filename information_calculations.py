"""
This package is a set of functions for analyzing the information
aspects of the trials. The actual running and plotting of the
analysis happens via a different script.
"""

import data_reader
import math, random
from collections import defaultdict
import matplotlib.pyplot as plt
import network_build as nb
from itertools import combinations


"""
Takes dictionaries of names and counts and turns it into a
KL divergence measure with the same supports
(see create_continuity_and_probabilities function)

It calculates the KL from Q, the 'seen' distribution
to P, the true distribution. It wants probability
distributions, i.e. not raw counts.
"""
def KL(seen_distribution, true_distribution):

  # Just double checking the distros have the same support
  assert len(seen_distribution) == len(true_distribution)

  # summing across the categories of the distribution
  KL = 0
  for key, Pi in true_distribution.items():
    Qi = seen_distribution[key]
    KL -= Pi * math.log(Qi/Pi)

  return KL


def JS(seen, true):
    """ Returns the JS divergence"""

    M={}
    for key, value in seen.items():
        M[key] = (value + true[key])/2


    JS1 = 0
    for key, p in seen.items():
        q = M[key]
        if q != 0 and p !=0:
            JS1 -= p*math.log(q/p)

    JS2 = 0
    for key, p in true.items():
        q = M[key]
        if q != 0 and p != 0:
            JS2 -= p*math.log(q/p)

    return JS1*.5 + JS2*.5


"""
A specialized function for forcing two distributions to have the same
support. It assumes the distribution with the more types has the correct
full support. In virtual of how the distributions are defined, it is not
possible for smaller of the two distributions to have names not in the
larger one.
"""
def create_continuity_and_probabilities(distro1, distro2):
    # figuring out which distribution should establish the support
    if len(distro1) > len(distro2):
        supporting_distro = dict(distro1)
        compared_distro = dict(distro2)
    else:
        supporting_distro = dict(distro2)
        compared_distro = dict(distro1)
    # finding which values are missing from the compared distribution (i.e. not supported)
    missing_keys_seen = list(set(supporting_distro.keys()).difference(set(compared_distro.keys())))
    missing_keys_true = list(set(compared_distro.keys()).difference(set(supporting_distro.keys())))

    if missing_keys_true != []:
        for ky in missing_keys_true:
            if ky in supporting_distro.keys():
                supporting_distro[ky] +=1
            else:
                supporting_distro[ky] = 1

    # the 'corrected' distribution assumes those missing names now all have a probability of .001
    new_compared_distro = {i:.001 for i in missing_keys_seen}

    # finding the number instances in the compared distribution
    instance_count = sum(compared_distro.values())

    # finding the number of categories with new weight of .001
    instances_missing = len(missing_keys_seen)

    """
    This loop corrects the compared distribution by re-weighting
    the original frequencies so that it is like they came from a
    sampling process in which the missing values were observed
    once in a thousand samples.
    """
    for ky, val in compared_distro.items():

        # true frequency
        frequency = val/float(instance_count)

        # count in 1000 value sample
        weighted_count = frequency*1000

        # subtracting for the missing instances in proportion to original weight
        new_count = weighted_count - (instances_missing * frequency)
        new_compared_distro[ky] = new_count/1000.


    """"Doing the same thing, but for the other distribution """
    new_supporting_distro = {i:.001 for i in missing_keys_true}

    # finding the number instances in the compared distribution
    instance_count = sum(supporting_distro.values())

    # finding the number of categories with new weight of .001
    instances_missing = len(missing_keys_true)

    for ky, val in supporting_distro.items():

        # true frequency
        frequency = val/float(instance_count)

        # count in 1000 value sample
        weighted_count = frequency*1000

        # subtracting for the missing instances in proportion to original weight
        new_count = weighted_count - (instances_missing * frequency)
        new_supporting_distro[ky] = new_count/1000.


    if set(new_compared_distro.keys()) != set(new_supporting_distro.keys()):
        print("not the same")
        print(distro1)
        print(distro2)


    return new_compared_distro, new_supporting_distro


"""
Real participants likely had limited recollection of the names seen.
To impose a very naive model of memory, impose_limited_memory() slices
down the full memory bank to a limited buffer. The memory_length parameter
defines the number of previous rounds the participants "use" for the
purposes of calculating KL divergences and information gain.

A second reason to do this is that the system doesn't need to remember
names that have "died off" and aren't in circulation anymore. Imposing
memory starts to solve this.

The default of a length of 25 is equivalent to full memory
"""

def impose_limited_memory(the_distro, memory_length=25):

    seen_distro_by_round = {}

    for game_round in range(1,26):
        last = 1 if game_round <= memory_length else game_round-memory_length
        new_list = []
        for theRound in range(last, game_round + 1):
            new_list.extend(the_distro[theRound])

        seen_distro_by_round[game_round] = list_to_dict_of_counts(new_list)

    return seen_distro_by_round



"""
A simple function for converting a raw list of
instances into a dictionary with the associated count.
The raw data comes as just a list.
"""
def list_to_dict_of_counts(theList):
    dct = defaultdict(int)
    for name in theList:
        dct[name] +=1
    return dct



"""
Part of the analytical approach is to consider the informational gain related to having
additional names. The actual name is a single instantiation of stochastic process, so we take a simulation approach, sampling from the possible names a participation could have seen many times and calculating the informational gainrelated to that new simulated "seen" distribution to get the bounds, average, and standard deviation of the information value of each additional name the
participant sees.
"""
def simulate_information_gains(seen_distro,
                                true_distro,
                                divergence_type="KL",
                                number_iterations = 200,
                                with_replacement = False,
                                graph = False,
                                figure_name = "simulation_test.png",
                                slope = True):

    # The baseline case is no simulated data, so we calculate the true
    # KL and standard dev (none) and put it in our lists of values.
    seen, true = create_continuity_and_probabilities(seen_distro, true_distro)
    true_KL = KL(seen, true)

    divergence_avg = [true_KL]

    # Now we find some information that we'll need inside the subsequent loop.
    # We'll use this list for randomly sampling names
    flat_list = []
    for key, value in true_distro.items():
        for i in range(value):
            flat_list.append(key)


    # Starting the simulution process: Given the participants have seen 2 names (theirs and their partner's) and there are a total of 24 participants, the maximum # of additional names is 22.
    for additional_names in range(1, 23):

        true_seen_distro = defaultdict(int)
        # Just copying over list of names actually seen.
        for key, value in seen_distro.items():
             true_seen_distro[key] = value

        within_divergences = []
        for i in range(number_iterations):
            simulated_seen_distro = true_seen_distro.copy()
            if with_replacement:
                additional_names_list = random.choices(flat_list, k=additional_names) # with replacement
            else:
                additional_names_list= random.sample(flat_list, additional_names) #without replacement

            for name in additional_names_list:
                simulated_seen_distro[name] += 1


            simulated_seen, true = create_continuity_and_probabilities(simulated_seen_distro, true_distro)

            if divergence_type == "KL":
                within_divergences.append(KL(simulated_seen, true))
            else:
                within_divergences.append(JS(simulated_seen, true))

        # Now that we simulated a bunch of different seen distributions, average them.
        avg = sum(within_divergences) / len(within_divergences)

        divergence_avg.append(avg)

    return divergence_avg


def simulate_weakest_ties(participant_data,
                            true_distro_by_round,
                            network_topology_name,
                            additional_name_count,
                            divergence_type="JS",
                            memory_length=5,
                            graph=False):
    """
    This method explores the informational value of signals from the "weakest links" and
    compares that value to the value of the additional homogeneous mixing names. To do this,
    we first get the "weakest", i.e. structurally distant, links for each node given the
    network topology. There can be multiple such links so we just sample the required number.
    """

    participant_differences = {}

    net = nb.net(network_topology_name)
    weakest_links = net.calc_weakest_ties(additional_name_count)

    # We start by going through each participant
    for participant, data in participant_data.items():
        # and we're going to calculate the total divergence for each weakest link

        # First we look at the overall divergence without unstructured names
        div_no_unstructured_by_round = []

        distro_by_round_no_unstructured = impose_limited_memory(data["distro_by_round_no_unstructured"],
                                                                memory_length=memory_length)
        for game_round in range(1,26):

            # a new list of names needs to be truncated to induce some memory
            seen_distro_with_no_unstructured, true_distro = create_continuity_and_probabilities(
                                                        distro_by_round_no_unstructured[game_round],
                                                        true_distro_by_round[game_round])

            # divergence between real and distro with additional names
            if divergence_type == "JS":
                div_no_unstructured_by_round.append(JS(seen_distro_with_no_unstructured, true_distro))
            else:
                div_no_unstructured_by_round.append(KL(seen_distro_with_no_unstructured, true_distro))


        # Next we calculate the divergence for the real rounds with unstructured
        div_with_unstructured_by_round = []

        distro_by_round_unstructured = impose_limited_memory(data["distro_by_round_unstructured"],
                                                                memory_length=memory_length)
        for game_round in range(1,26):

            # a new list of names needs to be truncated to induce some memory
            seen_distro_with_unstructured, true_distro_2 = create_continuity_and_probabilities(
                                                            distro_by_round_unstructured[game_round],
                                                            true_distro_by_round[game_round])
            # divergence between real and distro with additional names
            if divergence_type == "JS":
                div_with_unstructured_by_round.append(JS(seen_distro_with_unstructured, true_distro_2))
            else:
                div_with_unstructured_by_round.append(KL(seen_distro_with_unstructured, true_distro_2))

        names_to_add_by_round = defaultdict(list)
        for new_alter in weakest_links[participant]:
            # We get the name played by the partic. and append it to lists of names for each round
            for round, name in enumerate(participant_data[new_alter]["names_played"]):
                 names_to_add_by_round[round].append(name)

        # We go through round by round and add in the newly visible names
        simulated_distro_by_round = {}

        for game_round in range(1,26):
            no_unstructured = list(data["distro_by_round_no_unstructured"][game_round])
            no_unstructured.extend(names_to_add_by_round[game_round])
            simulated_distro_by_round[game_round] = no_unstructured

        # That new list of names needs to be truncated to induce some memory
        distro_by_round_weakest_link = impose_limited_memory(simulated_distro_by_round,
                                                             memory_length=memory_length)

        divergence_weakest_by_round = []
        for game_round in range(1,26):

            seen_distro_weakest_link, true_distro = create_continuity_and_probabilities(
                                            distro_by_round_weakest_link[game_round],
                                            true_distro_by_round[game_round])


            # Then we calculate the divergence from distro with weakest link to true distro
            if divergence_type == "JS":
                div_with_weakest_link = JS(seen_distro_weakest_link, true_distro)
            else:
                div_with_weakest_link = KL(seen_distro_weakest_link, true_distro)
            divergence_weakest_by_round.append(div_with_weakest_link)

        game_diffs = [i-j for i, j in zip(divergence_weakest_by_round,div_with_unstructured_by_round)]



        divergence_ratio = [1 - actual/total if total !=0 else 0 for actual, total in zip(div_with_unstructured_by_round, div_no_unstructured_by_round)]

        #print("comparison:", divergence_weakest_by_round)
        #print("base_diff:", div_no_unstructured_by_round)
        #print("ratio:", divergence_ratio)

        participant_differences[participant] = (game_diffs,
                                                weakest_links[participant],
                                                div_no_unstructured_by_round,
                                                divergence_ratio)

    return participant_differences


def find_non_neighbors_with_most_info(participant_data,
                                    true_distro_by_round,
                                    network_topology_name,
                                    additional_names_count,
                                    divergence_type="JS",
                                    memory_length=5,
                                    graph=False):
    """
    Rather than find the "weakest" ties like in simulate_weakest_ties, I just look at all
    non-neighbors and the information they possess relative to the ego. The ones with the most information become the baseline.

    When the additional names is more than one, the information yielded is calculated from the joint distribution of both (or all three) additional names. That means we need to look at all combos.
    """

    participant_differences = {}

    net = nb.net(network_topology_name)

    # We start by going through each participant
    for participant, data in participant_data.items():
        # and we're going to calculate the divergences for combo of additional names

        # First we look at the overall divergence without unstructured names
        div_no_unstructured_by_round = []

        distro_by_round_no_unstructured = impose_limited_memory(data["distro_by_round_no_unstructured"],
                                                                memory_length=memory_length)
        for game_round in range(1,26):

            # a new list of names needs to be truncated to induce some memory
            seen_distro_with_no_unstructured, true_distro = create_continuity_and_probabilities(
                                                        distro_by_round_no_unstructured[game_round],
                                                        true_distro_by_round[game_round])

            # KL divergence between real and distro with additional names
            if divergence_type == "JS":
                div_no_unstructured_by_round.append(JS(seen_distro_with_no_unstructured, true_distro))
            else:
                div_no_unstructured_by_round.append(KL(seen_distro_with_no_unstructured, true_distro))

        # Then we calculate the divergence for the real rounds with unstructured

        div_with_unstructured_by_round = []

        distro_by_round_unstructured = impose_limited_memory(data["distro_by_round_unstructured"],
                                                                memory_length=memory_length)
        for game_round in range(1,26):

            # a new list of names needs to be truncated to induce some memory
            seen_distro_with_unstructured, true_distro_2 = create_continuity_and_probabilities(
                                                            distro_by_round_unstructured[game_round],
                                                            true_distro_by_round[game_round])

            # KL divergence between real and distro with additional names
            if divergence_type == "JS":
                div_with_unstructured_by_round.append(JS(seen_distro_with_unstructured, true_distro_2))
            else:
                div_with_unstructured_by_round.append(KL(seen_distro_with_unstructured, true_distro_2))

        # We'll select the non-neighbors with the most information (the smallest diff in KL) so
        # we'll need to track the smallest divergences
        smallest_divergence = None

        # We get all the non-neighbors of the ego node
        non_neighbors = net.get_non_alters(participant)

        # We get all possible combinations of non-neighbors
        non_neighbor_combinations = combinations(non_neighbors,additional_names_count)


        for combo in non_neighbor_combinations:

            names_to_add_by_round = defaultdict(list)
            for not_neighbor in combo:
                # We get the names played by the participant and append it to lists of names for
                # each round
                for round, name in enumerate(participant_data[not_neighbor]["names_played"]):
                     names_to_add_by_round[round].append(name)

            # We go through round by round and add in the newly visible names
            simulated_distro_by_round = {}

            # now we add to distribution with out unstructured
            for game_round in range(1,26):
                no_unstructured = list(data["distro_by_round_no_unstructured"][game_round])
                no_unstructured.extend(names_to_add_by_round[game_round])

                simulated_distro_by_round[game_round] = no_unstructured

            # That new list of names needs to be truncated to induce some memory
            distro_by_round_with_non_neighbors = impose_limited_memory(simulated_distro_by_round,
                                                                    memory_length=memory_length)

            divergence_by_round = []
            for game_round in range(1,26):

                seen_distro_non_neighbors, true_distro = create_continuity_and_probabilities(
                                                distro_by_round_with_non_neighbors[game_round],
                                                true_distro_by_round[game_round])

                # Then we calculate the divergence from distro with weakest link to true distro
                if divergence_type == "JS":
                    div_with_non_neighbors = JS(seen_distro_non_neighbors, true_distro)
                else:
                    div_with_non_neighbors = KL(seen_distro_non_neighbors, true_distro)
                divergence_by_round.append(div_with_non_neighbors)


            # the sum is the area under the curve, or the total divergences for the whole run.
            total_diff_in_divergences = sum(divergence_by_round)

            if smallest_divergence != None:
                if total_diff_in_divergences < sum(smallest_divergence):
                    smallest_divergence = divergence_by_round
                    alters_with_smallest_divergence = combo
            else:
                smallest_divergence = divergence_by_round
                alters_with_smallest_divergence = combo

        # now that we have the smallest divergence to the true distro from simulated weakest
        # link distro, we compare it to the divergence to the true distro from the real additional
        # name distro. A positive difference as calculated means the additiona name version carries # more information (because it is closer to the true)

        game_diffs = [i-j for i, j in zip(smallest_divergence,div_with_unstructured_by_round)]

        divergence_ratio = [1 - actual/total if total !=0 else 0 for actual, total in zip(div_with_unstructured_by_round, div_no_unstructured_by_round)]

        #print("comparison:", divergence_by_round)
        #print("base_diff:", div_no_unstructured_by_round)
        #print("ratio:", divergence_ratio)

        participant_differences[participant] = (game_diffs, alters_with_smallest_divergence, div_no_unstructured_by_round, divergence_ratio)

    return participant_differences




def find_random_others(participant_data,
                        true_distro_by_round,
                        network_topology_name,
                        additional_names_count,
                        only_non_neighbors=False,
                        divergence_type="JS",
                        memory_length=5,
                        graph=False):
    """
    The basic baseline is a random sample of additional neighbors to see. We randomly select a combination of neighbors and calculate the difference in divergences.
    """

    participant_differences = {}

    net = nb.net(network_topology_name)

    # We start by going through each participant
    for participant, data in participant_data.items():
        # First we look at the overall divergence without unstructured names
        div_no_unstructured_by_round = []

        distro_by_round_no_unstructured = impose_limited_memory(data["distro_by_round_no_unstructured"],
                                                        memory_length=memory_length)
        for game_round in range(1,26):

            # a new list of names needs to be truncated to induce some memory
            seen_distro_with_no_unstructured, true_distro = create_continuity_and_probabilities(
                                                        distro_by_round_no_unstructured[game_round],
                                                        true_distro_by_round[game_round])

            # divergence between real and distro with no additional names
            if divergence_type == "JS":
                div_no_unstructured_by_round.append(JS(seen_distro_with_no_unstructured, true_distro))
            else:
                div_no_unstructured_by_round.append(KL(seen_distro_with_no_unstructured, true_distro))


        # Then we calculate the KL divergence for the real rounds with unstructured
        div_with_unstructured_by_round = []

        distro_by_round_unstructured = impose_limited_memory(data["distro_by_round_unstructured"],
                                                                memory_length=memory_length)
        for game_round in range(1,26):

            # a new list of names needs to be truncated to induce some memory
            seen_distro_with_unstructured, true_distro_2 = create_continuity_and_probabilities(
                                                            distro_by_round_unstructured[game_round],
                                                            true_distro_by_round[game_round])

            # KL divergence between real and distro with additional name
            if divergence_type == "JS":
                div_with_unstructured_by_round.append(JS(seen_distro_with_unstructured, true_distro_2))
            else:
                div_with_unstructured_by_round.append(KL(seen_distro_with_unstructured, true_distro_2))



        if only_non_neighbors == True:
            # We get all the non-neighbors of the ego node
            random_others = list(net.get_non_alters(participant))
        else:
            random_others = list(net.network.nodes())
            random_others.remove(participant)

        # We randomly select a combination of non-neighbors
        random_others_combination = random.sample(random_others, additional_names_count)

        names_to_add_by_round = defaultdict(list)
        for random_other in random_others_combination:
            # We get the names played by the participant and append it to lists of names for
            # each round
            for round, name in enumerate(participant_data[random_other]["names_played"]):
                 names_to_add_by_round[round].append(name)

        # We go through round by round and add in the newly visible names
        simulated_distro_by_round = {}

        # now we add to distribution with out unstructured
        for game_round in range(1,26):
            no_unstructured = list(data["distro_by_round_no_unstructured"][game_round])
            no_unstructured.extend(names_to_add_by_round[game_round])

            simulated_distro_by_round[game_round] = no_unstructured

        # That new list of names needs to be truncated to induce some memory
        distro_by_round_with_random_others = impose_limited_memory(simulated_distro_by_round,
                                                                memory_length=memory_length)

        divergence_by_round = []
        for game_round in range(1,26):

            seen_distro_random_others, true_distro = create_continuity_and_probabilities(
                                            distro_by_round_with_random_others[game_round],
                                            true_distro_by_round[game_round])

            # Then we calculate the divergence from distro with the additional names to true distro
            if divergence_type == "JS":
                div_with_random_others = JS(seen_distro_random_others, true_distro)
            else:
                div_with_random_others = KL(seen_distro_random_others, true_distro)
            divergence_by_round.append(div_with_random_others)

        # now that we have the divergence to the true distro from simulated distro, we compare it
        # to the divergence to the true distro from the real additional name distro. A positive
        # difference as calculated means the additional name version carries more information
        # (because it is closer to the true)

        game_diffs = [i-j for i, j in zip(divergence_by_round,div_with_unstructured_by_round)]

        divergence_ratio = [1 - actual/total if total !=0 else 0 for actual, total in zip(div_with_unstructured_by_round, div_no_unstructured_by_round)]

        #print("comparison:", divergence_by_round)
        #print("base_diff:", div_no_unstructured_by_round)
        #print("ratio:", divergence_ratio)

        participant_differences[participant] = (game_diffs, random_others_combination, div_no_unstructured_by_round, divergence_ratio)

    return participant_differences

def find_spare_alter(participant_data,
                            true_distro_by_round,
                            network_topology_name,
                            additional_names_count,
                            only_non_neighbors=False,
                            divergence_type="JS",
                            memory_length=5,
                            graph=False):
        """
        For this comparison, the participant is 'exposed' to a name played by a random alter the participant is not currently playing with.
        """

        participant_differences = {}

        net = nb.net(network_topology_name)

        # We start by going through each participant
        for participant, data in participant_data.items():
            # First we look at the overall divergence without unstructured names
            div_no_unstructured_by_round = []

            distro_by_round_no_unstructured = impose_limited_memory(data["distro_by_round_no_unstructured"],
                                                            memory_length=memory_length)
            for game_round in range(1,26):

                # a new list of names needs to be truncated to induce some memory
                seen_distro_with_no_unstructured, true_distro = create_continuity_and_probabilities(
                                                            distro_by_round_no_unstructured[game_round], true_distro_by_round[game_round])

                # divergence between real and distro with no additional names
                if divergence_type == "JS":
                    div_no_unstructured_by_round.append(JS(seen_distro_with_no_unstructured, true_distro))
                else:
                    div_no_unstructured_by_round.append(KL(seen_distro_with_no_unstructured, true_distro))


            # Then we calculate the KL divergence for the real rounds with unstructured
            div_with_unstructured_by_round = []

            distro_by_round_unstructured = impose_limited_memory(data["distro_by_round_unstructured"],
                                                                    memory_length = memory_length)

            for game_round in range(1,26):

                # a new list of names needs to be truncated to induce some memory
                seen_distro_with_unstructured, true_distro_2 = create_continuity_and_probabilities(
                                                            distro_by_round_unstructured[game_round],
                                                            true_distro_by_round[game_round])

                # KL divergence between real and distro with additional name
                if divergence_type == "JS":
                    div_with_unstructured_by_round.append(JS(seen_distro_with_unstructured, true_distro_2))
                else:
                    div_with_unstructured_by_round.append(KL(seen_distro_with_unstructured, true_distro_2))


            net = nb.net(network_topology_name)

            # here we get the fixed list of the participant's network alters
            alters = list(net.network.neighbors(participant))

            names_to_add_by_round = {}

            for round, round_partner in data["actual_alters"].items():
                # We randomly select additional alters until we have the right number that aren't the one the participant actually just played against. This is complicated in the case of the small world networks, which don't have a constant degree distribution. If there aren't enough alters for the required number of exposures, we get more from an alter's alter.
                list_of_random_alters = []
                new_alters = list(alters)
                new_alters.remove(round_partner)
                if len(new_alters) >= additional_names_count:
                    list_of_random_alters = random.sample(new_alters, additional_names_count)
                else:
                    more_needed = additional_names_count - len(new_alters)
                    next_alter = 0
                    alters_to_add = new_alters
                    while more_needed > 0:
                        alters_alters = list(net.network.neighbors(new_alters[next_alter]))
                        try:
                            alters_alters.remove(round_alter)
                        except:
                            pass
                        try:
                            alters_alters.remove(participant)
                        except:
                            pass
                        alters_to_add.extend(alters_alters)
                        next_alter +=1
                        more_needed -= len(alters_to_add)
                    # It's possible that we now have too many names, so we truncate the list
                    list_of_random_alters = alters_to_add[:additional_names_count]
                    assert len(list_of_random_alters) == additional_names_count, (list_of_random_alters, additional_names_count)



                # We get the name the random alter played and add it to the list
                other_names = []
                for other_alter in list_of_random_alters:
                    other_names.append(participant_data[other_alter]["names_played"][round-1])
                names_to_add_by_round[round] = other_names


            # We go through round by round and add in the newly visible names
            simulated_distro_by_round = {}

            # now we add to distribution with out unstructured
            for game_round in range(1,26):
                no_unstructured = list(data["distro_by_round_no_unstructured"][game_round])
                no_unstructured.extend(names_to_add_by_round[game_round])

                simulated_distro_by_round[game_round] = no_unstructured

            # That new list of names needs to be truncated to induce some memory
            distro_by_round_with_random_others = impose_limited_memory(simulated_distro_by_round,
                                                                    memory_length=memory_length)

            div_with_random_by_round = []
            for game_round in range(1,26):

                seen_distro_random_others, true_distro = create_continuity_and_probabilities(
                                                distro_by_round_with_random_others[game_round],
                                                true_distro_by_round[game_round])

                # Then we calculate the divergence from distro with the additional names to true distro
                if divergence_type == "JS":
                    div_with_random_others = JS(seen_distro_random_others, true_distro)
                else:
                    div_with_random_others = KL(seen_distro_random_others, true_distro)
                div_with_random_by_round.append(div_with_random_others)

            # now that we have the divergence to the true distro from simulated distro, we compare it
            # to the divergence to the true distro from the real additional name distro. A positive
            # difference as calculated means the additional name version carries more information
            # (because it is closer to the true)

            game_diffs = [i-j for i, j in zip(div_with_random_by_round,div_with_unstructured_by_round)]

            divergence_ratio = [1 - actual/total if total !=0 else 0 for actual, total in zip(div_with_unstructured_by_round, div_no_unstructured_by_round)]

            #print("comparison:", div_with_random_by_round)
            #print("base_diff:", div_no_unstructured_by_round)
            #print("ratio:", divergence_ratio)

            participant_differences[participant] = (game_diffs, list_of_random_alters, div_no_unstructured_by_round, divergence_ratio)

        return participant_differences
