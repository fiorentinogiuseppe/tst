import pickle
import gensim

with open('pages_text_original.pickle', 'rb') as handle:
    machado_sents = pickle.load(handle)


# Tamanho do 'embedding'
N = 200

# Número de palavras anteriores a serem consideradas
C = 7

model = gensim.models.Word2Vec(machado_sents, sg=0, size=N, window=C, min_count=5, hs=0, negative=14)
model.save("machado_model.model")

