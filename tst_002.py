from gensim.models import KeyedVectors

def word_embedding(word):
    return model.wv[word]

# Pega apenas as palavras a partir do resultado da função 'most_similar'
def strip_score(result):
    return [w for w, s in result]

# Lista as palavras mais próximas
def closest_words(word, num=5):
    word_score_pair = model.wv.most_similar(word, topn=num)
    return strip_score(word_score_pair)


model = KeyedVectors.load("machado_model.model", mmap='r')

test_words = ['seja', 'foi', 'amou', 'aquele', 'foram', 'homem', 'rua', 'marcela']

for w in test_words:
    print(w, closest_words(w))