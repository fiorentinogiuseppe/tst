from gensim.models import KeyedVectors
import json
from natas import ocr_builder


def load_json(file_path):
    text = {}
    with open(file_path,) as outfile:
        text = json.load(outfile)
    return text


model = KeyedVectors.load("machado_model.model", mmap='r')
seed_words = load_json('/home/giuseppe/PycharmProjects/natas_spellcheck/test/counts_1grams.txt')
res = ocr_builder.extract_parallel(seed_words, model, min_frequency=1000,  lemmatize=True, use_freq=True)
print(res)
