"""
This module rebuilds the network from a text file of the round by round pairings.
In principle, I should have saved the adjacency matrix, but apparently I didn't. The reason
is likely that the round by round pairings contain that information, but also moreself.

"""

import re, random
import networkx as nx
from collections import defaultdict


class net():

    def __init__(self, topology_name):
        """
        Initialization recreates the network as a NetworkX graph object and
        also saves the round by round pairings
        """

        self.network = nx.Graph()
        self.alter_pairings_by_round = defaultdict(list)

        with open("../experiment_data/network_topologies/{}".format(topology_name),"r") as f:
            for line in f.readlines():
                if "round_number" not in line:
                    matches = re.findall(r"\(\d*,\s?\d*\)",line) #regex fits something like (12,13)
                    for pair in matches:
                        pr = [int(i) for i in pair.replace("(","").replace(")","").split(",")]
                        # There are many nodes in the network that don't contribute to the game
                        # dynamics. They exist to hold extras/spots that got blocked out by Otree.
                        if all([i<24 for i in pr]):
                            self.network.add_edge(pr[0],pr[1])
                            self.alter_pairings_by_round[pr[0]].append(pr[1])
                            self.alter_pairings_by_round[pr[1]].append(pr[0])



    def get_network(self):

        return self.network

    def get_pairings_by_round(self):

        return self.alter_pairings_by_round

    def select_weakest_tie(self, lengths):

        length_set = list(set(lengths.values()))
        length_set.sort(reverse=True)

        # get all alters with a shortest path equal to the maximum length
        max_alters = [int(alter) for alter, length in lengths.items() if length==length_set[0]]

        return random.choice(max_alters)



    def calc_weakest_ties(self, additional_names_count):
        """
        To assess the value of additional names at random, I compare it to the value of the names from the "weakest" ties, that is a new link between the ego any of the nodes with the longest shortest pair to the ego. This link will be a bridging tie in the sense that it shortens distances, but it need not alter the centralities of the network much.

        We need as many weakest_ties as the additional_names_count variable, but the "weakness" of the subsequent ties depends on the selection of the prior ties. This means to select the truly weakest set of ties, we need to recalculate the shortest paths each time.
        """
        weakest_ties = {}


        # The nx call gives all the shortest paths between all pairs in the network
        for values in nx.all_pairs_shortest_path_length(self.network):
            # the all_pairs_shortest_path method returns a generator which has keys of the source
            # node, and a dictionary of paths to targets, keyed by the target
            source = values[0]
            paths_to_targets = values[1]

            first_weakest = self.select_weakest_tie(paths_to_targets)
            this_nodes_weakest = [first_weakest]

            # now we do it again until we have enough names
            if additional_names_count > 1:
                # first we create a copy of the network to modify safely
                graph_copy = self.network.copy()
                # we add the new weakest tie
                graph_copy.add_edge(source, first_weakest)

                # then we recalculate the paths
                new_paths = nx.single_source_shortest_path_length(graph_copy, source)
                second_weakest=self.select_weakest_tie(nx.single_source_shortest_path_length(graph_copy, source))
                this_nodes_weakest.append(second_weakest)

                # One graph had three names, so we need to do it one more time
                if additional_names_count > 2:
                    graph_copy.add_edge(source, second_weakest)
                    new_paths = nx.single_source_shortest_path_length(graph_copy, source)
                    third_weakest=self.select_weakest_tie(nx.single_source_shortest_path_length(graph_copy, source))
                    this_nodes_weakest.append(third_weakest)

            weakest_ties[source] = this_nodes_weakest
        return weakest_ties

    def get_non_alters(self, ego):

        return nx.non_neighbors(self.network, ego)
