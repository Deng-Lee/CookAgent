import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

db_path = "data/chroma"
collection_name = "cook_chunks"
chunk_id = "素炒豆角-3-f1ef074d"

client = chromadb.PersistentClient(path=db_path)
collection = client.get_collection(name=collection_name)
res = collection.get(ids=[chunk_id])

print(res["documents"][0])
print(res["metadatas"][0])
