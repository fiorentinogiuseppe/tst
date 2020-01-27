import gensim
import codecs
import unidecode
import re
import nltk

def strip_score(result):
    return [w for w, s in result]


def load_text(file_path):
    text = []
    with codecs.open(file_path, encoding="utf-8-sig") as f:
        for line in f:
            text.append(line)
    return text


def closest_words(word, num=5):
    word_score_pair = model.wv.most_similar(word, topn=num)
    return strip_score(word_score_pair)

machado_sents = load_text("../data/Text/OCR/pages_text.txt")
text = ' '.join(machado_sents)
text_cleared = re.sub('\s+', ' ', text)
text_cleared_lower = text_cleared.lower()
text_cleared_lower_sem = unidecode.unidecode(text_cleared_lower)

corpus = [[word.lower() for word in text_cleared_lower_sem.split()]]
print(corpus)


model = gensim.models.Word2Vec(corpus, min_count=1, size=32)
print(model.wv.vocab)
