from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

s1 = "The car is very fast."
s2 = "That vehicle has great speed."

embeddings = model.encode([s1, s2])
score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

print(f"S1: {s1}")
print(f"S2: {s2}")
print(f"Similarity: {score:.4f}")