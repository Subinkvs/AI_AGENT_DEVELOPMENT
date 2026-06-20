from sklearn.feature_extraction.text import TfidfVectorizer

'''TF-IDF Vectorizer is primarily a keyword-based retrieval method that ranks documents 
using word importance scores.'''

# Sample documents
docs = [
    "office equipment policy",
    "office furniture guidelines",
    "office travel policy"
]

# Create a TF-IDF vectorizer object
vectorizer = TfidfVectorizer()

# Learn vocabulary and compute TF-IDF scores
tfidf_matrix = vectorizer.fit_transform(docs)

# Get the vocabulary words (column names)
feature_names = vectorizer.get_feature_names_out()

# Convert sparse matrix to a NumPy array for easy viewing
scores = tfidf_matrix.toarray()

# Iterate through each document and its corresponding TF-IDF row
for doc_num, (doc, row) in enumerate(zip(docs, scores), start=1):

    # Print the current document
    print(f"\nDocument {doc_num}: {doc}")

    # Pair each vocabulary word with its TF-IDF score
    for word, score in zip(feature_names, row):

        # Print the word and its TF-IDF score
        print(f"{word:<12}: {score:.6f}")


# Function to find document containing the query
def get_document_id(query):
    for index, doc in enumerate(docs):
        if query in doc:
            return index
    return -1


query = "travel"

query_scores = vectorizer.transform([query])

print(f"\nQuery Scores: {query_scores.toarray()}")

doc_id = get_document_id(query)

if doc_id != -1:
    print(f"Query: {query} -> Document {doc_id + 1}")
    print(f"Matched Document: {docs[doc_id]}")
else:
    print("No matching document found")