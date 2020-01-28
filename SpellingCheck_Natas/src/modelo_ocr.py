import multiprocessing
from gensim.models import Word2Vec
from mikatools import *
import re
from unidecode import unidecode
import spacy
from gensim.models.phrases import Phrases, Phraser
from collections import defaultdict

nlp = spacy.load('pt')


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

def clear(text):
    text = ' '.join(text)
    semacento = unidecode(text)
    text_cleared = re.sub('[^A-Za-z0-9 ]+', '', semacento)
    text_cleared_lower = text_cleared.lower()
    doc = nlp(text_cleared_lower)
    new = [[token.lemma_ for token in doc if not(token.is_stop or token.is_punct)]]
    #corpus = [[word.lower() for word in text_cleared_lower.split()]]
    return new

#print("Load text")
#ground_truth = load_text("../data/Text/Original/pages_text_original.txt")
print("Load ocr_text.w2v")
train = load_text("../data/Text/OCR/pages_text.txt")

#print("Clearing text")
#ground_truth_cleared = clear(ground_truth)
print("Clearing ocr")
train_cleared = clear(train)

sent = train_cleared
phrases = Phrases(sent, min_count=30, progress_per=10000)
bigram = Phraser(phrases)

sentences = bigram[sent]
word_freq = defaultdict(int)
for sent in sentences:
    for i in sent:
        word_freq[i] += 1
cores = multiprocessing.cpu_count()
w2v_model = Word2Vec(min_count=1,
                     window=2,
                     size=300,
                     sample=6e-5,
                     alpha=0.03,
                     min_alpha=0.0007,
                     negative=20,
                     workers=cores-1)

w2v_model.build_vocab(sentences, progress_per=10000)
w2v_model.init_sims(replace=True)
w2v_model.save("../model/ocr_text.w2v")