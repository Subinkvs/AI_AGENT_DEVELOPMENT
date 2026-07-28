from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np

'''Embeddings are numerical vector representations of data (text, images, audio, etc.) 
that capture meaning and relationships.'''

sentences = [
    "I have a Dog",
    "I have a Dog"
]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

embeddings = model.encode(sentences)

similarity = np.dot(
    embeddings[0],
    embeddings[1]
)

print(f"{similarity * 100:.2f}% similar")




