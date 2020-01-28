import codecs
import gensim
from mikatools import script_path, json_load
from gensim.test.utils import common_texts
from natas import ocr_builder
from natas import ocr_correct_words

def load(file_path):
    text = []
    with codecs.open(file_path, encoding="utf-8-sig") as f:
        for line in f:
            text.append(line)
    return text


ocr_text = load("../data/Text/OCR/pages_text.txt")
#original_text = load("../data/Text/Original/pages_text_original.txt")
text = ' '.join(ocr_text)
corpus = [[word.lower() for word in text.split()]]

#with open('/home/giuseppe/PycharmProjects/natas_project/src/wiktionary_lemmas.json') as json_file:
#    wiktionary = json.load(json_file)

wiktionary = set([x.lower() for x in json_load(script_path("/wiktionary_lemmas.json"))])

# train model
model = gensim.models.Word2Vec(corpus, min_count=1)
print(model.wv.most_similar(positive=['triste']))

dictionary = wiktionary #Lemmas of the English Wiktionary, you will need to change this if working with any other language
lemmatize = True #Uses Spacy with English model, use natas.set_spacy(nlp) for other models and languages
seed_words = ["triste"]
print(seed_words)
results = ocr_builder.extract_parallel(seed_words, model, dictionary=dictionary, lemmatize=lemmatize)
print(results)



"""from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
# define training data
sentences = corpus
# train model
model = Word2Vec(sentences, min_count=1)
# fit a 2d PCA model to the vectors
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()"""