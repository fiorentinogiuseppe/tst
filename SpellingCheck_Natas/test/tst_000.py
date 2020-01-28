import codecs
import json
from natas import ocr_builder
from mikatools import *
from gensim.models import Word2Vec
import re
import multiprocessing



def load_text(file_path):
    text = []
    with codecs.open(file_path, encoding="utf-8-sig") as f:
        for line in f:
            text.append(line)
    return text


def load_json(file_path):
    text = {}
    with open(file_path,) as outfile:
        text = json.load(outfile)
    return text


print("Load ocr_text.w2v")
ocr_text = load_text("../data/Text/OCR/pages_text.txt")
print("Load original_text")
original_text = load_text("../data/Text/Original/pages_text_original.txt")
print("Load json")
seed_words = load_json('/home/giuseppe/PycharmProjects/natas_spellcheck/test/counts_1grams.txt')
seed_words = set(open_read("words-in-dictionary").read().split("\n"))

print("Creating text")
text = ' '.join(ocr_text)
print("Clearing text")
text_cleared = re.sub(r'[\!"#$%&\*+,-./:;<=>?@^_`()|~=]', '', text)
text_cleared_lower = text_cleared.lower()
print("Corpus")
corpus = [[word.lower() for word in text_cleared_lower.split()]]
print("Model")
model = Word2Vec(corpus, min_count=1)

#print("Extract parallel")
res = ocr_builder.extract_parallel(seed_words, model, min_frequency=1000,  lemmatize=False, use_freq=True)
print(res)

cores = multiprocessing.cpu_count()

w2v_model = Word2Vec(min_count=20,
                     window=2,
                     size=300,
                     sample=6e-5,
                     alpha=0.03,
                     min_alpha=0.0007,
                     negative=20,
                     workers=cores-1)
