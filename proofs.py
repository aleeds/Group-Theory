from groups import WhiteheadGraph
from groups import GetGens
from groups import WhiteheadAut
from groups import CyclicallyReduce
from groups import GensFromGraph

# This function will determine if the Whitehead Graph of a given word is
# Strongly Symmetric. It will do this using a theorem from the paper which
# states that a graph is strongly symmetric iff it is connected.
# Therefore this algorithm will simply determine if the graph is connected.
# This algorithm has been tested against all of the examples from the paper
def SSymmetric(word):
    graph = WhiteheadGraph(word)
    return SSymmetricGraph(graph)

def SSymmetricGraph(graph):
    gen = graph[0][0]
    visit = [gen]
    # Classic queue based visit all nodes
    for i in visit:
        for (a,b) in graph:
            if i == a and b not in visit:
                visit += b
            if i == b and a not in visit:
                visit += a
    gens = GensFromGraph(graph)
    for g in gens:
        if g not in visit:
            return False
    return True

# This function will return a tuple of lists of characters. The first shall be
# the generators that gen can reach, the second shall be the list (possibly
# empty) of generators that gen cannot reach

def Separate(word,gen):
  visit = [gen]
  gens = GetGens(word)
  graph = WhiteheadGraph(word)
  for i in visit:
    for (a,b) in graph:
      if i == a and b not in visit:
        visit += b
      if i == b and a not in visit:
        visit += a
  nv = []
  for g in gens:
    if g not in visit:
      nv += g

  return (visit,nv)
# This function will take a word, and return the word in a strongly symmetric
# form. In addition, with a special flag, it will return the sequence of
# Whitehead automorphisms which resulted in this new word. This implements the
# proof from the paper. Future updates may have it return the automorphism as a
# single automorphism TODO(aleeds)
# re = '0' will return nothing extra
# re = '1' will return the sequence
# re = '2' will return the automorphism once implemented
# re = '3' will return both
def MakeSS(word,re = '0'):
    gens = GetGens(word)
    auts = []
    for g in gens:
        (z,zc) = Separate(word,g)
        if zc != []:
            aut = (g,z)
            word = WhiteheadAut(word,aut)
            auts += aut
            gens = GetGens(word)
    if re == '0':
        return CyclicallyReduce(word)
    elif re == '1':
        return (word,auts)
    # add other cases

# This function will take a word,and a generator, and return true if it is a cut
# point of the Whitehead Graph.
def CutPoint(word,gen):
    graph = WhiteheadGraph(word)
    gens = GetGens(word)
    graph_minus = []
    for (a,b) in graph:
        if a != gen and b != gen:
            graph_minus += [(a,b)]
    # I am not happy with this hack. It basically just makes sure every
    # generator is in the graph, so it works when I check for connectivity
    for g in gens:
        if g != gen:
            graph_minus += [(g,g)]
    return not SSymmetricGraph(graph_minus)

####
"""
I need to flesh out these comments, and add more return options for the functions
"""
####

# This will remove a Cutpoint from the word
def RemoveCutPoint(word,gen):
    print("Not Completed")
    # Needs to return the automorphism
    return word

def NoCutPointsSS(word):
    gens = GetGens(word)
    for g in gens:
        if CutPoint(word,g):
            word = RemoveCutPoint(word,g)
            if not SSymmetric(word):
                word = MakeSS(word)
                gens = GetGens(word)
    # Add stuff so it returns the automorphisms if it wants, like MakeSS
    return word
