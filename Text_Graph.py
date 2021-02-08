
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

print(stopwords.words('english'))
sent = "We find prominent gas hydrate provinces offshore Central America where sediments are rich in organic carbon and in the Arctic Ocean where low bottom water temperatures stabilize methane hydrates. The world’s total gas hydrate inventory is estimated at 0.82x1013 m3–2.10x1015 m3 CH4 (at STP conditions) or, equivalently, 4.18–995 Gt of methane carbon. The first value refers to present day conditions estimated using the relatively low Holocene sedimentation rates; the second value corresponds to a scenario of higher Quaternary sedimentation rates along continental margins."

stop_words = set(stopwords.words('english'))

word_tokens = word_tokenize(sent)

filtered_sentence = [w for w in word_tokens if not w in stop_words]

filtered_sentence = []

for w in word_tokens:
    if w not in stop_words:
        filtered_sentence.append(w)

print(word_tokens)
print(filtered_sentence)