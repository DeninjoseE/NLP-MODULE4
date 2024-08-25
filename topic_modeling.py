import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Step 1: Load the dataset
newsgroups = fetch_20newsgroups(subset='all', categories=['alt.atheism', 'comp.graphics', 'sci.med', 'sci.space'])
documents = newsgroups.data

# Step 2: Preprocess the text data
stop_words = stopwords.words('english')
vectorizer = CountVectorizer(stop_words=stop_words)
doc_term_matrix = vectorizer.fit_transform(documents)

# Step 3: Prepare for LDA with Gensim
tokenized_docs = [doc.split() for doc in documents]
dictionary = corpora.Dictionary(tokenized_docs)
corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

# Create Gensim LDA model
lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=4, random_state=42, passes=10)

# Display the discovered topics
topics = lda_model.print_topics()
for idx, topic in topics:
    print(f"Topic {idx + 1}: {topic}")

# Step 4: LDA Visualization with pyLDAvis
lda_display = gensimvis.prepare(lda_model, corpus, dictionary, sort_topics=False)

# If using Jupyter Notebook or similar, display the visualization inline
pyLDAvis.display(lda_display)

# Alternatively, save the visualization to an HTML file
pyLDAvis.save_html(lda_display, 'lda_visualization.html')

# Step 5: Word Embeddings for Document Similarity
word2vec_model = Word2Vec(sentences=tokenized_docs, vector_size=100, window=5, min_count=2, workers=4)

# Represent documents as averaged word vectors
def document_vector(doc):
    doc = [word for word in doc if word in word2vec_model.wv.key_to_index]
    return np.mean(word2vec_model.wv[doc], axis=0) if len(doc) > 0 else np.zeros(100)

doc_vectors = [document_vector(doc) for doc in tokenized_docs]

# Step 6: Document Similarity and Clustering
similarity_matrix = cosine_similarity(doc_vectors)

kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(similarity_matrix)

# Visualization of clusters
plt.scatter(similarity_matrix[:, 0], similarity_matrix[:, 1], c=kmeans.labels_)
plt.title('Document Clusters based on Similarity')
plt.show()
