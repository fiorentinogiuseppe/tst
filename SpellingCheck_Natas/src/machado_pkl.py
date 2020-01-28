import pickle

from mikatools import *

import unicodedata
from nltk.corpus import machado


def strip_accents_and_lower(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn').lower()

machado_sents = map(lambda sent: list(map(strip_accents_and_lower, sent)), machado.sents())
machado_sents = list(machado_sents)

with open("../data/Sentences/pages_text_original.pickle", "wb") as handle:
    pickle.dump(machado_sents, handle)

