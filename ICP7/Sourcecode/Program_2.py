# final script

import requests
from bs4 import BeautifulSoup
import nltk

nltk.download('averaged_perceptron_tagger')
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk import wordpunct_tokenize, pos_tag, ne_chunk

nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('punkt')
nltk.download('wordnet')

########## EXTRACT DATA FROM URL ############
web_page_url = 'https://en.wikipedia.org/wiki/Google'
# Issue a simple HTTP request to get the webpage text
google_page = requests.get(web_page_url)
google_page_soup = BeautifulSoup(google_page.text, 'html.parser')

########## TOKENIZATION #############
# we need string in order to apply tokenization
google_page_text = google_page_soup.get_text()  # get_text() gives us string
s_tokens = nltk.sent_tokenize(google_page_text)
w_tokens = nltk.word_tokenize(google_page_text)

######### POS #############
# apply POS on word tokens obtained from above
pos_text = nltk.pos_tag(w_tokens)

######### STEMMING ############
# apply stemming
p_stemmer = PorterStemmer()
porter_stem = []
lancaster_stem = []
snowball_stem = []

for word in w_tokens:
    porter_stem.append(p_stemmer.stem(word))
    # print('word: {} || porter stem:{}'.format(word, p_stemmer.stem(word)))

# apply Lancaster and snowball in a similar manner
lan_stemmer = LancasterStemmer()
for word in w_tokens:
    lancaster_stem.append(lan_stemmer.stem(word))
    # print('word: {} || lancaster stem:{}'.format(word, lan_stemmer.stem(word)))

snow_stemmer = SnowballStemmer('english')
for word in w_tokens:
    snowball_stem.append(snow_stemmer.stem(word))
    # print('word: {} || snowball stem:{}'.format(word, snow_stemmer.stem(word)))

########## LAMMETIZATION ###########
# apply lammetization
lemmatizer = WordNetLemmatizer()
lemma = []
for word in w_tokens:
    lemma.append(lemmatizer.lemmatize(word))
    # print('word:{} lemma:{}'.format(word,lemmatizer.lemmatize(word)))

########## NAMED ENTITY RECOGNITION #############
# apply Named Entity Recognition
NER = []
for sentence in s_tokens:
    NER.append(ne_chunk(pos_tag(wordpunct_tokenize(sentence))))
    # print("sentence: {} \n\n NER: {} \n\n".format(sentence, ne_chunk(pos_tag(wordpunct_tokenize(sentence)))))

######### TRI GRAMS ###########
# convert to trigrams
from nltk.util import ngrams

n = 3  # for trigrams
char_trigram = []

# ngrams over words
for word in w_tokens:
    trigrams = (ngrams(word, n))
    for tri in trigrams:
        # print(tri)
        char_trigram.append(tri)

# ngrams over sentences
word_trigram = []


def get_ngrams(text, n):
    n_grams = ngrams(nltk.word_tokenize(text), n)
    return [' '.join(grams) for grams in n_grams]


for sentence in s_tokens:
    word_trigram.append(get_ngrams(sentence, 3))

######### OUTPUTS ###########

print('SENTENCE TOKENIZATION of extracted text: {}'.format(s_tokens[:20]))
print('\n\n\n')
print('WORD TOKENIZATION of extracted text: {}'.format(w_tokens[:20]))
print('\n\n\n')
print('POS output: {}'.format(pos_text[:20]))
print('\n\n\n')
print('Porter Stemming output: {}'.format(porter_stem[:20]))
print('\n\n\n')
print('Lancaster Stemming output: {}'.format(lancaster_stem[:20]))
print('\n\n\n')
print('Snowball Stemming output: {}'.format(snowball_stem[:20]))
print('\n\n\n')
print('LEMMATIZATION OUTPUT : {}'.format(lemma[:20]))
print('\n\n\n')
print('NER OUTPUT : {}'.format(NER[:20]))
print('\n\n\n')
print('CHAR TRIGRAM OUTPUT : {}'.format(char_trigram[:20]))
print('\n\n\n')
print('WORD TRIGRAM OUTPUT : {}'.format(word_trigram[:20]))

# As the output was huge I limited it to first 20 lines