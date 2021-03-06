import codecs
import json
from natas import ocr_builder
from mikatools import *
from gensim.models import Word2Vec
import re
from unidecode import unidecode


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

print("Creating text")
text = ' '.join(ocr_text)

print("Clearing text")
text_cleared = re.sub(r'[\!"#$%&\*+,-./:;<=>?@^_`()|~=]', '', text)
text_cleared_lower = unidecode(text_cleared.lower())

print("Corpus")
corpus = list(set([[word.lower() for word in text_cleared_lower.split()]][0]))

print("Model")
#model = Word2Vec(corpus
# , min_count=1)
model = Word2Vec.load('../model/machado_model.model')

res = ocr_builder.extract_parallel(corpus, model, use_freq=False)
print(res)
#print(model.wv.vocab)
#print(original_text)

#https://hackernoon.com/neural-machine-translation-using-open-nmt-for-training-a-translation-model-1129a3a2a2d3
