# This file will use a new format for representing automorphisms.
# An automorphism (if it is polynomially growing) will can be represented
# as an n-tuple (if it lives in F_n). This is how they will be represented.

# For example, ["","a","baB"] is the automorphism:
#     [("a","a"),("b","ba"),("c","cbaB"]]

from groups import ApplyGenAut

# This function will convert from the previously described form into the
# normal, assuming that gens is the order that you want the generators in.
def ConvertToNormalForm(poly_form,gens = ['a','b','c','d','e','f','g','h']):
    ret = []
    for i in range(0,len(poly_form)):
        ret += [(gens[i],gens[i] + poly_form[i])]
    return ret

# Here will go ConvertToPolyForm if neccesary

def ApplyPoly(word,poly,gens = ['a','b','c','d','e','f','g','h']):
  normal = ConvertToNormalForm(poly,gens)
  return ApplyGenAut(word,normal)
