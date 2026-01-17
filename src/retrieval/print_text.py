import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
# "羊肉汤-title", "羊肉汤-1-53e41191", "羊肉汤-2-9a0456c5"
db_path = "data/chroma"
collection_name = "cook_chunks"
chunk_id = "羊肉汤-2-9a0456c5"

client = chromadb.PersistentClient(path=db_path)
collection = client.get_collection(name=collection_name)
res = collection.get(ids=[chunk_id])

print(res["documents"][0])
print(res["metadatas"][0])
