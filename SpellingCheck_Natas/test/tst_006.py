import codecs
import gensim
from mikatools import script_path, json_load, open_read, json_dump
from gensim.test.utils import common_texts
from natas import ocr_builder
from natas import ocr_correct_words
from natas import ocr_builder
import spacy
import natas

def load(file_path):
    text = []
    with codecs.open(file_path, encoding="utf-8-sig") as f:
        for line in f:
            text.append(line)
    return text


#ocr_text.w2v = load("../data/Text/OCR/pages_text.txt")
original_text = load("../data/Text/Original/pages_text_original.txt")
text = ' '.join(original_text)
corpus = [[word.lower() for word in text.split()]]

#with open('/home/giuseppe/PycharmProjects/natas_project/src/wiktionary_lemmas.json') as json_file:
#    wiktionary = json.load(json_file)

dictionary = set([x.lower() for x in json_load(script_path("/home/giuseppe/PycharmProjects/natas_project/src/wiktionary_lemmas.json"))])

# train model
model = gensim.models.Word2Vec(corpus, min_count=1)
nlp = spacy.load("pt_core_news_sm")
natas.set_spacy(nlp)

results = ocr_builder.extract_parallel(seed_words, model, dictionary=dictionary)
