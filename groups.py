from graphs import GraphEquality

# Produces WhiteheadGraph for the word s


def WhiteheadGraph(s):
    ret = []
    for i in range(0, len(s)):
        r = (InverseChar(s[i]), s[(i + 1) % len(s)])
        ret.append(r)
    return ret

# Gets the generators for a list of words


def GetGens(words):
    gens = []
    for word in words:
        for q in word:
            if (q.lower() not in gens):
                gens.append(q.lower())
    return gens

# Produces WhiteheadGraph for words sl


def WhiteheadGraphList(sl):
    ret = []
    for s in sl:
        for i in WhiteheadGraph(s):
            if (not cont(i, ret)):
                ret.append(i)
    return ret

# gets the valence of x in Whit


def Valence(x, Whit):
    c = 0
    for i in Whit:
        if (i[0] == x or i[1] == x):
            c += 1
    return c

# Gets the valence of x in the WhiteheadGraph of word


def ValenceWord(x, word):
    return Valence(x, WhiteheadGraph(word))

# applies whitehead Aut (x,Z) to x


def ApplyWhiteAut(x, Z, e):
    s = [WhiteheadAutG(x, Z, c) for c in e]
    r = ""
    for i in s:
        r = add(r, i)
    return r

# Applies Aut to all words in sl


def ChangeWords(sl, x, Z):
    return [ApplyWhiteAut(x, Z, i) for i in sl]

# Inverses, 'a' -> 'A', 'b' -> 'B', etc


def InverseChar(a):
    if (a.lower() == a):
        return a.upper()
    else:
        return a.lower()

# Says if y \elem Z


def cont(y, Z):
    return Z.count(y) > 0

# Gets the inverse of a word


def Inverse(x):
    rx = x[::-1]
    return ''.join([InverseChar(r) for r in rx])

# Defines Aut for one character words


def WhiteheadAutG(x, Z, y):
    if (InverseChar(y) == x):
        return Inverse(WhiteheadAutG(x, Z, Inverse(y)))
    if (y == x):
        return x
    elif (not cont(y, Z) and not cont(InverseChar(y), Z)):
        return y
    elif (cont(y, Z) and not cont(InverseChar(y), Z)):
        return x + y
    elif (not cont(y, Z) and cont(InverseChar(y), Z)):
        return y + InverseChar(x)
    else:
        return x + y + InverseChar(x)


def PowerSet(elems):
    if (len(elems) == 0):
        return []
    elif (len(elems) == 1):
        return [[], elems]
    else:
        head, *tail = elems
        p = PowerSet(tail)
        return [[head] + i for i in p] + p


def PowerSetSize(elems, size):
    return [x for x in PowerSet(elems) if (len(x) == size)]


def ReduceGraph(word, gens):
    auts = GenerateAllWhiteheadAuts(gens)
    t = [WhiteheadAut(word, whitehead) for whitehead in auts]
    return [x for x in t if len(WhiteheadGraph(x)) < len(WhiteheadGraph(word))]


def GetGraphAndReduceWord(word, gens):
    auts = GenerateAllWhiteheadAuts(gens)
    t = [WhiteheadAut(word, whitehead) for whitehead in auts]
    return [WhiteheadGraph(x) for x in t if len(WhiteheadGraph(x)) < len(WhiteheadGraph(word))]


def GetReducingWhiteheadAut(word, gens):
    auts = GenerateAllWhiteheadAuts(gens)
    t = [(WhiteheadAut(word, w), w) for w in auts]
    return [w for (x, w) in t if len(WhiteheadGraph(x)) < len(WhiteheadGraph(word))]


def ReduceGraphList(words, gens):
    auts = GenerateAllWhiteheadAuts(gens)
    t = [[WhiteheadAut(word, w) for word in words] for w in auts]
    return [x for x in t if len(WhiteheadGraphList(x)) < len(WhiteheadGraphList(words))]


def ReduceToBottom(word, gens):
    first = [word]
    h = first
    ret = []
    while (h != []):
        ret = h
        t = []
        for i in h:
            t += ReduceGraph(i, gens)
        h = t
    return ret

def GetUniques(ls):
    ret = []
    for i in ls:
        if i not in ret:
            ret.append(i)
    return ret

def ReduceWords(words,gens):
    first = ReduceGraphList(words,gens)
    ret = []
    tmp = first
    i = 0
    while (i < len(tmp)):
        t = ReduceGraphList(tmp[i],gens)
        if (t == []):
            ret.append(tmp[i])
        else:
            t = GetUniques(t)
            tmp += t
        i += 1
    return GetUniques(ret)


def GenerateAllWhiteheadAuts(gens):
    ret = []
    big_gens = gens + [InverseChar(i) for i in gens]
    for x in big_gens:
        gensTwo = [i for i in big_gens if x != InverseChar(i)]
        power_set = PowerSet(gensTwo)
        for s in power_set:
            if (cont(x, s)):
                ret.append((x, s))
    return ret


# Applies whitehead automorphism Aut to x
def WhiteheadAut(x, Aut):
    return ApplyWhiteAut(Aut[0], Aut[1], x)

# Applies the automorphism which switches gen and its inverse


def ApplyInverseAut(word, gen):
    ret = ""
    for i in word:
        if (i == gen or i == InverseChar(gen)):
            ret += InverseChar(i)
        else:
            ret += i
    return ret

# Applies permutations to word


def ApplyPermutationAut(word, perms):
    ret = ""
    for c in word:
        for (a, b) in perms:
            if (a == c):
                ret += b
            elif (a == InverseChar(c)):
                ret += InverseChar(b)
    return ret


def ProduceAllWhiteheadGraphs(n, gens):
    gensMod = gens + [InverseChar(i) for i in gens]
    pairs = [(i, q) for i in gensMod for q in gensMod if (i != q)]
    graph = []
    for i in pairs:
        if i not in graph and (i[1], i[0]) not in graph:
            graph.append(i)
    return PowerSetSize(graph, n)


def add(a, b):
    ra = a[::-1]
    c = 0
    for i in range(0, min(len(a), len(b))):
        if (InverseChar(ra[i]) == b[i]):
            c += 1
        else:
            break
    ret = ""
    for i in range(0, len(a) - c):
        ret += a[i]
    for i in range(c, len(b)):
        ret += b[i]
    return ret


def ReducedWhiteheads(words, gens):
    smallest_words = ReduceWords(words, gens)
    return [WhiteheadGraphList(words) for words in smallest_words]


def InvertTup(t, gen):
    return (ApplyInverseAut(t[0], gen), ApplyInverseAut(t[1], gen))


def PermuateTup(t, perm):
    return (ApplyPermutationAut(t[0], perm), ApplyPermutationAut(t[1], perm))


def ConnectingInverse(graph_one, graph_two, gens):
    for gen in gens:
        a = [InvertTup(tup, gen) for tup in graph_two]
        if GraphEquality(a, graph_one):
            return True
    return False


def GenerateAllPermutations(gens):
    from itertools import permutations
    perms = list(permutations(gens))
    ret = []
    for perm in perms:
        ret.append([(gens[i], perm[i]) for i in range(0, len(gens))])
    return ret


def ConnectingPermutation(graph_one, graph_two, gens):
    perms = GenerateAllPermutations(gens)
    for perm in perms:
        a = [PermuateTup(tup, perm) for tup in graph_two]
        if GraphEquality(graph_one, a):
            return True
    return False


def ConnectingAut(graph_one, graph_two, gens):
    t = ConnectingInverse(graph_one, graph_two, gens)
    s = ConnectingPermutation(graph_one, graph_two, gens)
    return s or t


def DrawGraphForWord(word):
    g = WhiteheadGraph(word)
    gens = GetGens(word)
    import networkx as nx
    import matplotlib.pyplot as plt
    import numpy
    G = nx.MultiGraph()
    G.add_edges_from(g)
    pos = nx.circular_layout(G)
    #nx.draw(G,pos)
    labels = {}
    for gen in gens:
        nx.draw_networkx_nodes(G,pos,nodelist=[gen,InverseChar(gen)],
                               node_color=numpy.random.rand(3,1))
        labels[gen] = gen
        labels[InverseChar(gen)] = InverseChar(gen)
    nx.draw_networkx_edges(G,pos)
    nx.draw_networkx_labels(G,pos,labels)
    plt.axis('off')
    plt.show()

def DrawGraphForWords(words):
    g = WhiteheadGraphList(words)
    import networkx as nx
    import matplotlib.pyplot as plt
    G = nx.MultiGraph()
    G.add_edges_from(g)
    nx.draw_networkx_nodes(G)
    plt.show()
