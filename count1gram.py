# write all code in one cell#========================Load data=========================
import codecs
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
import re
import json

def load(file_path):
    text = []
    with codecs.open(file_path, encoding="utf-8-sig") as f:
        for line in f:
            text.append(line)
    return text


def create_frequency_table(text_string) -> dict:

    stopWords = set(stopwords.words("portuguese"))
    words = nltk.word_tokenize(text_string)
    ps = PorterStemmer()

    freqTable = dict()
    for word in words:
        word = ps.stem(word)
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    return freqTable

original_text = load("../data/Text/OCR/pages_text.txt")

# convert string to lower case
text = ' '.join(original_text)
text_cleared = re.sub('\s+', ' ', text)
text_cleared_lower = text_cleared.lower()
freqTab = create_frequency_table(text_cleared_lower)

file_name = 'counts_1grams.txt'
json = json.dumps(freqTab, indent=4, ensure_ascii=False)
f = open(file_name,"w")
f.write(json)
f.close()


