from groups import *
import graphs as graphs


def FindGraphPosition(g, all_graphs):
    i = 0
    for k in all_graphs:
        if GraphEquality(k, g):
            break
        i += 1
    return i


def ProduceMetaGraph(all_graphs,gens):
    graph_of_graphs = []
    for i in range(len(all_graphs)):
        for q in range(i, len(all_graphs)):
            if (i != q and ConnectingAut(all_graphs[i], all_graphs[q], gens)):
                graph_of_graphs.append((i, q))
    return graph_of_graphs

# Make the whitehead graphs less complex. If they don't match, then it
#   is False
# Make a list of all graphs of that size
# ProduceMetaGraph
# Find if two graphs are in the same section. If there are two graphs in the
# same section, then true. If not, false


def WhiteheadAlgorithm(words_one, words_two):
    gens = GetGens(words_one + words_two)
    print("Gens")
    graphs_one = ReducedWhiteheads(words_one, gens)
    print("Graph_one")
    graphs_two = ReducedWhiteheads(words_two, gens)
    if (len(graphs_one[0]) != len(graphs_two[0])):
        return False
    size = len(graphs_one[0])
    print(size)
    all_graphs = ProduceAllWhiteheadGraphs(size, gens)
    print(len(all_graphs))
    graph_of_graphs = ProduceMetaGraph(all_graphs,gens)
    print(len(graph_of_graphs))
    for iV in graphs_one:
        for qV in graphs_two:
            i = FindGraphPosition(iV, all_graphs)
            q = FindGraphPosition(qV, all_graphs)
            if (graphs.PathConnected(i, q, graph_of_graphs)):
                return True
    return False


def WhiteheadSingle(word):
    gens = GetGens(word)
    return any([WhiteheadAlgorithm(word, gen) for gen in gens])

def WhiteheadWords(words):
    gens = GetGens(words)
    
