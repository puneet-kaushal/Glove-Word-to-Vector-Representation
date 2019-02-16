from glove import Corpus, Glove
from sklearn.decomposition import PCA  # put this at the top of your program
#to plot the graph using matplotlib 
import matplotlib.pyplot as plt
#load the glove model
glove = Glove.load('glove.model')
#vectors for each word 
word2vec = glove.word_vectors
#dictionary of the word
dictionary = glove.dictionary
print()
print('dictionary',dictionary)
print()
#finding the one dimensional array for word hard
array=  glove.word_vectors[glove.dictionary['hard']]
print(word2vec.ndim)

font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 10,
        }
#x and y coordinates where you need to display the text.
plt.text(0.5, -0.06, r'Word "hard" in vector form', fontdict=font)
plt.plot(array)
plt.grid(True)

#finding the one dimensional array for word best
array2 = glove.word_vectors[glove.dictionary['best']]
plt.text(2.1, 0.050, r'Word "best" in vector form', fontdict=font)
plt.plot(array2)
plt.ylabel('Word Embedding', fontsize=12, color='red')
plt.xlabel('X label', fontsize=12, color='red')
plt.title(r'Vectors of words Glove',color='red')
plt.show()
