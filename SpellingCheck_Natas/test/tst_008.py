import codecs
import gensim
import spacy
from natas import ocr_builder
from unidecode import unidecode
import re
from nltk.tokenize import word_tokenize
from natas.normalize import  call_onmt


nlp = spacy.load('pt')
def load(file_path):
    text = []
    with codecs.open(file_path, encoding="utf-8-sig") as f:
        for line in f:
            text.append(line)
    return text

def clear(text):
    text = ' '.join(text)
    semacento = unidecode(text)
    text_cleared = re.sub('[^A-Za-z0-9 ]+', '', semacento)
    text_cleared_lower = text_cleared.lower()
    doc = nlp(text_cleared_lower)
    new = [[token.lemma_ for token in doc if not(token.is_stop or token.is_punct)]]
    return new

#original_text = set(clear(load("../data/Text/Original/pages_text_original.txt"))[0])

# train model
model = gensim.models.Word2Vec.load("/home/giuseppe/tmp/aiboxsummerschool-OCR/py/projects/SpellingCheck_Natas/model/ocr_text.w2v")

seed_words = "Guardei a carta e o relógio, e esperei a filosofia"
doc = nlp(seed_words)
new = set([unidecode(token.lemma_.lower()) for token in doc if not(token.is_stop or token.is_punct)])

results = ocr_builder.extract_parallel(new, model, use_freq=False, lemmatize=False, word_len=2)
ocr = dict()
for key_dict, value_dict in results.items():
    tmp_list =[]
    for key, value in value_dict.items():
        if value <= 3:
            lista = ' '.join([l for l in key])
            tmp_list.append(lista)
    if len(tmp_list) > 0:
        ocr.update({' '.join([l for l in key_dict]):tmp_list})

print(ocr)