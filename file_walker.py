import os
from collections import defaultdict
from scipy import stats

from data_reader import *
from information_calculations import *



memory_length = 10

for directory, sub, files in os.walk("../experiment_data"):
    for current_file_name in files:
        if "Data" in current_file_name and not "ignore" in current_file_name and not "0addtl-" in current_file_name:
            import_file_for_MLM("/".join([directory,current_file_name]))
            """"
            # extracting the right data from the Otree CSV
            game_data = import_file("/".join([directory,current_file_name]))

            # packing the game data in useful ways
            participant_data, group_round_names = preanalysis_packing(game_data)

            # Applying "limited memory" to the whole system to discount names no longer in circulation
            true_distro_by_round = impose_limited_memory(group_round_names, memory_length)



            This first routine does a few things. First it calculates each participant's
            difference in divergences. That is, the difference between the divergences of the participant's seen name distribution both with and without the additional stigmergy names. This is one way to assess the informational value of the additional names.

            To put this unscaled value in context, we also calculate the average of each participant's total divergence from the distribution without additional names to the true distribution. This is roughly the ground to be "made up" enroute to having full information.

            average_total_KL = defaultdict(list)
            group_differences_by_round = defaultdict(list)


            for participant, data in participant_data.items():

                difference_list = []
                # First we impose a memory limit on the seen distribution
                distro_by_round_no_stigmergy = impose_limited_memory(
                                                            data["distro_by_round_no_stigmergy"],  memory_length=memory_length)
                distro_by_round_stigmergy = impose_limited_memory(
                                                            data["distro_by_round_stigmergy"],  memory_length=memory_length)


                for game_round in range(1,26):
                    # Next we create a seen distro with same support as true distro
                    seen_distro_no_stigmergy, true_distro = create_continuity_and_probabilities(
                                                        distro_by_round_no_stigmergy[game_round],
                                                        true_distro_by_round[game_round])

                    # Then we calculate the divergence
                    KL_no_stigmergy = KL(seen_distro_no_stigmergy, true_distro)


                    # Same as above, but for the distribution with the additional name
                    seen_distro_with_stigmergy, true_distro_2 = create_continuity_and_probabilities(
                                                        distro_by_round_stigmergy[game_round],
                                                        true_distro_by_round[game_round])



                    KL_with_stigmergy = KL(seen_distro_with_stigmergy, true_distro_2)

                    # difference is information gained by having the stigmergy names.
                    diff = KL_no_stigmergy - KL_with_stigmergy
                    difference_list.append(diff)
                    group_differences_by_round[game_round].append(diff)

                    # tracking the total divergence
                    average_total_KL[game_round].append(KL_no_stigmergy)

                # This plots the series of differences for a single participant
                plt.plot(difference_list)


            # Plotting the average total divergence with blue dashes
            plt.plot([sum(average_total_KL[game_round])/25 for game_round in range(1,26)],"b+")


            plt.axis([0,24,-1,4.5])
            plt.tight_layout()
            plt.savefig("../analysis/individual_KL_differences_"+current_file_name.replace(".csv",".png"),dpi=300)
            plt.clf()


            group_average = []
            group_std_dev_up = []
            group_std_dev_down = []

            for game_round, values in group_differences_by_round.items():
                average = sum(values)/25
                group_average.append(average)
                std_dev = (sum([(i-average)**2 for i in values])/25)**.5
                group_std_dev_up.append(average+std_dev)
                group_std_dev_down.append(average-std_dev)

            plt.plot(group_average, color="k")
            plt.plot(group_std_dev_up,"k--")
            plt.plot(group_std_dev_down,"k--")
            plt.plot([sum(average_total_KL[game_round])/25 for game_round in range(1,26)],"b+")
            plt.axis([0,24,-1,4.5])


            plt.savefig("../analysis/average_KL_differences_"+current_file_name.replace(".csv",".png"),dpi=300)

            plt.clf()




            divergence_type = "JS"
            simulation_type = "random_alter"
            only_non_neighbors = True

            file_stem = current_file_name.split("-")
            network_topology_name = file_stem[-1].replace(".csv",".txt")
            additional_names_count = int(file_stem[-2].replace("stig",""))

            if simulation_type == "weakest":
                results = simulate_weakest_ties(participant_data,
                                            true_distro_by_round,
                                            network_topology_name,
                                            additional_names_count,
                                            divergence_type=divergence_type,
                                            memory_length=memory_length)

            elif simulation_type == "most_info":
                results = find_non_neighbors_with_most_info(participant_data,
                                                        true_distro_by_round,
                                                        network_topology_name,
                                                        additional_names_count,
                                                        divergence_type=divergence_type,
                                                        memory_length=memory_length)
            elif simulation_type == "random_others":

                results = find_random_others(participant_data,
                                                        true_distro_by_round,
                                                        network_topology_name,
                                                        additional_names_count,
                                                        only_non_neighbors=only_non_neighbors,
                                                        divergence_type=divergence_type,
                                                        memory_length=memory_length)

            elif simulation_type == "random_alter":

                results = find_spare_alter(participant_data,
                                                        true_distro_by_round,
                                                        network_topology_name,
                                                        additional_names_count,
                                                        only_non_neighbors=only_non_neighbors,
                                                        divergence_type=divergence_type,
                                                        memory_length=memory_length)

            else:
                raise ValueError("Incorrect simulation type")


            differences_by_round = defaultdict(list)
            div_to_true_by_round = defaultdict(list)
            for list_of_diffs, alter, div_no_stigmergy_by_round, ratio in results.values():
                for round, value in enumerate(list_of_diffs):
                    differences_by_round[round].append(value)
                for round, value in enumerate(div_no_stigmergy_by_round):
                    div_to_true_by_round[round].append(value)


            averages_by_round = [stats.tmean(round) for round in differences_by_round.values()]
            sem_dev_by_round = [stats.tsem(round) for round in differences_by_round.values()]

            sem_below_by_round = [i-j for i, j in zip(averages_by_round,sem_dev_by_round)]
            sem_above_by_round = [i+j for i, j in zip(averages_by_round,sem_dev_by_round)]

            x_range= range(1,26)
            plt.plot(x_range, averages_by_round, color="k")
            plt.plot(x_range, sem_below_by_round,"k--")
            plt.plot(x_range, sem_above_by_round,"k--")

            if divergence_type == "JS":
                plt.axis([1,25,-.5,1])
            else:
                plt.axis([1,25,-.5,4.5])

            plt.hlines(0,1,25, "k")
            total_div_averages_by_round = [stats.tmean(round) for round in div_to_true_by_round.values()]
            total_div_sem_dev_by_round = [stats.tsem(round) for round in div_to_true_by_round.values()]

            total_div_sem_below_by_round = [i-j for i, j in zip(total_div_averages_by_round,total_div_sem_dev_by_round)]
            total_div_sem_above_by_round = [i+j for i, j in zip(total_div_averages_by_round,total_div_sem_dev_by_round)]

            plt.plot(x_range, total_div_averages_by_round, color="b")
            plt.plot(x_range, total_div_sem_below_by_round,"b--")
            plt.plot(x_range, total_div_sem_above_by_round,"b--")

            if only_non_neighbors == True:
                plt.savefig("../analysis/{}_diffs_{}_mem_{}_non_neighbors_".format(divergence_type, simulation_type, memory_length)+current_file_name.replace(".csv",".png"),dpi=300)
            else:
                plt.savefig("../analysis/{}_diffs_{}_mem_{}_".format(divergence_type, simulation_type, memory_length)+current_file_name.replace(".csv",".png"),dpi=300)

            plt.clf()

            #for alter, series in results.items():
            #    print(series)
            #    plt.plot(series[1])
            #plt.axis([0,24,-1,4.5])
            #plt.tight_layout()
            #plt.savefig("../analysis/weakest_link_differences_"+current_file_name.replace(".csv",".png"),dpi=300)
            #plt.clf()



            
            for rnd in range(1,26):
              group_slopes = []
              for participant, data in participant_data.items():

                  slopes = simulate_information_curve(data["distro_by_round_nostig"][rnd],true_distro_by_round[rnd])
                  group.append(slopes[0])
            #"""
