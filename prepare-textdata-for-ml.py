from sklearn.feature_extraction.text import CountVectorizer
# list of text documents
text = ["The quick brown fox jumped over the lazy dog."]
text2 = ["the puppy"]
t3 = ["It was very encouraging to know that funds for our Digital Library project have been released\
and project is on the track once again. I wish to propose the following fpr utilizing this\
opportunity to fetch best possible results."]
# create the transform
vectorizer = CountVectorizer()
# tokenize and build vocab
vectorizer.fit(text)
# vectorizer.fit(text2)
# summarize
print(vectorizer.vocabulary_)
# encode document
vector = vectorizer.transform(text)
vector1 = vectorizer.transform(text2)
vector2 = vectorizer.transform(t3)
# summarize encoded vector
print(vector.shape)
print(type(vector))
print(vector.toarray())

print(vector1.shape)
print(type(vector1))
print(vector1.toarray())

print(vector2.shape)
print(type(vector2))
print(vector2.toarray())