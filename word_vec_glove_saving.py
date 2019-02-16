#importing the glove libray which is developed by the standford university
from glove import Corpus, Glove
#lines whose words representation we need
lines=["Hello, Hope you study hard",
        "I believe, this is the best things we can do",
        "We are going to office"]
#variable to append the words of each index of the list
new_lines=[]
for line in lines:
    line=line.split(' ')
    new_lines.append(line)
#this is the corpus object
corpus = Corpus()
corpus.fit(new_lines, window=10)

#this will reprsent the word with length=5 for each word.
glove = Glove(no_components=5, learning_rate=0.05)
#model is trained for 30 epochs
glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
#dictionary is added
glove.add_dictionary(corpus.dictionary)
#model is saved.
glove.save('glove.model')
