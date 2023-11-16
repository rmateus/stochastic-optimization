from collections import defaultdict
import numpy as np


# Stochastic Graph class
class StochasticGraph:
    def __init__(self):
        self.nodes = list()
        self.edges = defaultdict(list)
        self.lower = {}
        self.distances = {}
        self.upper = {}

    def add_node(self, value):
        self.nodes.append(value)

    # create edge with uniform distributed weight
    def add_edge(self, from_node, to_node, lower, upper):
        self.edges[from_node].append(to_node)
        self.distances[(from_node, to_node)] = 1
        self.lower[(from_node, to_node)] = lower
        self.upper[(from_node, to_node)] = upper

    # return the expected length of the shortest paths w.r.t. given node
    def bellman(self, target_node):
        """
        Computes the shortest path from the target_node to all other nodes in the graph using the Bellman-Ford algorithm.

        Parameters:
        target_node (int): The node from which to compute the shortest path to all other nodes.

        Returns:
        dict: A dictionary containing the shortest distance from the target_node to each node in the graph.
        """
        inflist = [np.inf] * len(self.nodes)
        # vt - value list at time t for all the nodes w.r.t. to target_node
        vt = {k: v for k, v in zip(self.nodes, inflist)}
        vt[target_node] = 0

        # decision function for nodes w.r.t. to target_node
        dt = {k: v for k, v in zip(self.nodes, self.nodes)}

        # updating vt
        for t in range(1, len(self.nodes)):
            for v in self.nodes:
                for w in self.edges[v]:
                    # Bellman's equation
                    if vt[v] > vt[w] + 0.5 * (self.lower[(v, w)] + self.upper[(v, w)]):
                        vt[v] = vt[w] + 0.5 * (self.lower[(v, w)] + self.upper[(v, w)])
                        dt[v] = w
        # print(vt)
        # print(g.distances)
        return vt

    def truebellman(self, target_node):
        """
        Compute the value function for each node in the graph with respect to a given target node.

        Parameters:
        target_node (int): The target node for which the Bellman value is computed.

        Returns:
        tuple: A tuple containing the node with the largest distance to the target node and the associated distance.
        """
        inflist = [np.inf] * len(self.nodes)
        # vt - list for values at time t for all the nodes w.r.t. to target_node
        vt = {k: v for k, v in zip(self.nodes, inflist)}
        vt[target_node] = 0

        # decision function for nodes w.r.t. to target_node
        dt = {k: v for k, v in zip(self.nodes, self.nodes)}

        # updating vt
        for t in range(1, len(self.nodes)):
            for v in self.nodes:
                for w in self.edges[v]:
                    # Bellman equation
                    if vt[v] > vt[w] + self.distances[(v, w)]:
                        vt[v] = vt[w] + self.distances[(v, w)]
                        dt[v] = w

        # Find the node with the largest distance to the target node and the associated distance
        v_aux = {k: -1 if v == np.inf else v for k, v in vt.items()}
        max_node = max(v_aux, key=v_aux.get)
        max_dist = v_aux[max_node]

        return (max_node, max_dist)


def randomgraphChance(prng, n, p, LO_UPPER_BOUND, HI_UPPER_BOUND):
    g = StochasticGraph()
    for i in range(n):
        g.add_node(str(i))
    # Randomly create edges between nodes
    for i in range(n):
        for j in range(n):
            q = prng.uniform(0, 1)
            if i != j and q < p:
                # Every edge has a random weight between lo and hi
                lo = prng.uniform(0, LO_UPPER_BOUND)
                hi = prng.uniform(lo, HI_UPPER_BOUND)
                g.add_edge(str(i), str(j), lo, hi)
    return g


def printFormatedDict(dictInput):
    nodeList = [int(node) for node in dictInput.keys()]
    nodeList = sorted(nodeList)

    for node in nodeList:
        print("\t\tkey_{} = {:.2f}".format(str(node), dictInput[str(node)]))


def createStochasticGraph(params):
    # create a random graph of n nodes and make sure there is a feasible path from node '0' to node 'n-1'
    prng = np.random.RandomState(params["seed"])
    g = randomgraphChance(
        prng,
        params["nNodes"],
        params["probEdge"],
        params["LO_UPPER_BOUND"],
        params["HI_UPPER_BOUND"],
    )
    print("Created the graph")

    maxSteps = 0
    max_origin_node = None
    max_target_node = None
    # Determine two nodes that have the max number of edges between them.
    # These will be our origin and target (to avoid trivial or infeasible problems)
    for target_node in g.nodes:
        max_node, max_dist = g.truebellman(target_node)

        if max_dist > maxSteps:
            maxSteps = max_dist
            max_origin_node = max_node
            max_target_node = target_node

    print(
        "max_origin_node: {} -  max_target_node: {}  - distance: {}".format(
            max_origin_node, max_target_node, maxSteps
        )
    )

    # This will be our initial guess
    V_0 = g.bellman(max_target_node)

    print("Computed V_0")
    print(printFormatedDict(V_0))

    return g, V_0, max_origin_node, max_target_node, maxSteps
