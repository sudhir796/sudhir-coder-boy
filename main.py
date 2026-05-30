from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Define the document
document='''AI is transforming education.
RAG helps reduce halucination in LLMs.
Embedding converts text into vectors'''
print("Original Document:")
print(document)
print("-" * 30)

# 2. Split the document into chunks
chunks = [chunk.strip() for chunk in document.split('\n') if chunk.strip()]
print("Chunks:")
print(chunks)
print("-" * 30)

# 3. Load the SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 4. Encode the chunks
chunk_embeddings = model.encode(chunks)
print(f"Shape of chunk embeddings: {chunk_embeddings.shape}")
print("-" * 30)

# 5. Define the query
query = "what reduces halucinations"
print(f"Query: {query}")
print("-" * 30)

# 6. Encode the query
query_embedding = model.encode(query)
print(f"Shape of query embedding: {query_embedding.shape}")
# print(f"Query embedding: {query_embedding}") # Optional: to see the full embedding
print("-" * 30)

# 7. Calculate cosine similarities
similarities = cosine_similarity(
    [query_embedding],
    chunk_embeddings
)
print(f"Similarities: {similarities}")
print("-" * 30)

# 8. Find the best matching chunk
best_index = similarities.argmax()
retrieved_chunks = chunks[best_index]
print(f"Best retrieved chunk: {retrieved_chunks}")
print("-" * 30)

# 9. Construct the prompt
prompt = f'''
Context:
{retrieved_chunks}
Question:
{query}
'''
print("Generated Prompt:")
print(prompt)