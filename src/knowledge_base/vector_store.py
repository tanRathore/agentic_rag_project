# src/knowledge_base/vector_store.py
import os
import pandas as pd
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
import config 
import uuid

class VectorDB:
    def __init__(self):
        print("VectorDB __init__ called.")
        self.client = None
        self.embedding_model = None

        try:
            self.client = QdrantClient(url=config.QDRANT_URL)
            print(f"Attempting to connect to Qdrant at URL: {config.QDRANT_URL}")
            self.client.get_collections() 
            print(f"Successfully connected to Qdrant at {config.QDRANT_URL}")
        except Exception as e:
            print(f"ERROR: Could not connect to Qdrant at {config.QDRANT_URL}. Error: {e}")
            self.client = None 

        if self.client: 
            try:
                print(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}...")
                self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
                print(f"Embedding model '{config.EMBEDDING_MODEL_NAME}' loaded successfully.")
            except Exception as e:
                print(f"ERROR: Could not load embedding model '{config.EMBEDDING_MODEL_NAME}'. Error: {e}")
                self.embedding_model = None 
    
    def is_ready(self):
        """Checks if both client and embedding model are initialized."""
        return self.client is not None and self.embedding_model is not None

    def create_collection(self, collection_name=config.QDRANT_COLLECTION_NAME, vector_size=config.VECTOR_SIZE):
        """Creates or recreates a new collection in Qdrant."""
        if not self.client:
            print("ERROR: Qdrant client not initialized. Cannot create collection.")
            return False
        try:
            self.client.recreate_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
            )
            print(f"Collection '{collection_name}' created/recreated successfully with vector size {vector_size}.")
            return True
        except Exception as e:
            print(f"ERROR: Creating collection '{collection_name}'. Error: {e}")
            return False

    def get_collection_info(self, collection_name=config.QDRANT_COLLECTION_NAME):
        """Gets information about a collection."""
        if not self.client:
            print("ERROR: Qdrant client not initialized. Cannot get collection info.")
            return None
        try:
            return self.client.get_collection(collection_name=collection_name)
        except Exception as e:
            if "not found" in str(e).lower() or "404" in str(e).lower():
                print(f"Collection '{collection_name}' not found.")
            else:
                print(f"ERROR: Getting collection info for '{collection_name}'. Error: {e}")
            return None

    def embed_text(self, text_or_texts):
        """Embeds a single text or a list of texts."""
        if not self.embedding_model:
            print("ERROR: Embedding model not loaded. Cannot embed text.")
            return None
        if not text_or_texts:
            return [] if isinstance(text_or_texts, list) else None
            
        # print(f"Embedding text(s). Input type: {type(text_or_texts)}, length: {len(text_or_texts) if isinstance(text_or_texts, list) else 1}")
        try:
            embeddings = self.embedding_model.encode(text_or_texts, show_progress_bar=False) 
            return embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings
        except Exception as e:
            print(f"ERROR: During text embedding. Error: {e}")
            return None

    def upsert_points_batch(self, collection_name: str, points_df: pd.DataFrame, batch_size: int = 64):
        """
        Embeds descriptions and upserts data points to Qdrant in batches.
        Expects 'description' column in points_df for embedding,
        and 'id' column for point IDs. Other columns are used as payload.
        """
        if not self.is_ready():
            print("ERROR: VectorDB not ready (client or model missing). Cannot upsert points.")
            return 0
        if 'description' not in points_df.columns:
            print("ERROR: 'description' column not found in DataFrame. Cannot embed.")
            return 0
        if 'id' not in points_df.columns:
            print("ERROR: 'id' column not found in DataFrame. Point IDs are required.")
            return 0

        num_points_upserted = 0
        total_points_to_process = len(points_df)
        records = points_df.to_dict(orient='records')

        for i in range(0, total_points_to_process, batch_size):
            batch_records = records[i:i + batch_size]
            current_batch_size = len(batch_records)
            descriptions_batch = [str(record.get('description', '')) for record in batch_records]
            
            print(f"Processing batch {i//batch_size + 1}/{(total_points_to_process -1)//batch_size + 1}: Embedding {current_batch_size} descriptions...")
            embeddings_batch = self.embed_text(descriptions_batch)

            if embeddings_batch is None or len(embeddings_batch) != current_batch_size:
                print(f"ERROR: Embedding failed or returned incorrect number of embeddings for batch starting at index {i}. Skipping batch.")
                continue

            qdrant_points = []
            for record_idx, (record, embedding) in enumerate(zip(batch_records, embeddings_batch)):
                payload = {}
                for key, value in record.items():
                    if pd.isna(value):
                        payload[key] = None
                    elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
                        payload[key] = str(value)
                    else:
                        payload[key] = value
                
                try:
                    point_id_val = record['id']
                    if not isinstance(point_id_val, (int, str)): 
                        try:
                            point_id_val = int(point_id_val)
                        except ValueError:
                            print(f"Warning: Point ID {point_id_val} is not int/str and couldn't be cast to int. Converting to string for record at original index {i+record_idx}.")
                            point_id_val = str(point_id_val)
                    
                    qdrant_points.append(
                        PointStruct(
                            id=point_id_val,
                            vector=embedding,
                            payload=payload
                        )
                    )
                except Exception as e:
                    print(f"Error creating PointStruct for record ID {record.get('id', 'N/A')} (original index {i+record_idx}): {e}. Skipping this point.")

            if qdrant_points:
                try:
                    self.client.upsert(
                        collection_name=collection_name,
                        points=qdrant_points,
                        wait=True 
                    )
                    num_points_upserted += len(qdrant_points)
                    print(f"Successfully upserted batch. Total points now upserted in this run: {num_points_upserted}")
                except Exception as e:
                    print(f"ERROR: Upserting batch to Qdrant. Error: {e}")
            
        print(f"Finished upserting. Total points processed: {total_points_to_process}, successfully upserted in this run: {num_points_upserted}")
        return num_points_upserted

    def search_points(self, query_text: str, collection_name: str = config.QDRANT_COLLECTION_NAME, top_k: int = 5):
        """
        Searches for points in the collection based on the similarity of their vectors to the query text's embedding.
        """
        if not self.is_ready():
            print("ERROR: VectorDB not ready. Cannot perform search.")
            return []
        if not query_text:
            print("Warning: Empty query text received for search.")
            return []

        print(f"Searching in collection '{collection_name}' for query: '{query_text[:100]}...' with top_k={top_k}")
        try:
            query_embedding = self.embed_text(query_text)
            if query_embedding is None:
                print("ERROR: Failed to embed query text.")
                return []

            search_results = self.client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=top_k,
                with_payload=True
            )
            print(f"Search returned {len(search_results)} results.")
            return search_results
        except Exception as e:
            print(f"ERROR: During Qdrant search for query '{query_text[:50]}...'. Error: {e}")
            return []

if __name__ == '__main__':
    print("Running vector_store.py directly for testing...")
    db = VectorDB()
    if db.is_ready():
        print("\nVectorDB initialized successfully for testing.")
        test_collection_name = "test_direct_run_collection"
        db.create_collection(collection_name=test_collection_name, vector_size=config.VECTOR_SIZE)
        info = db.get_collection_info(collection_name=test_collection_name)
        if info: print(f"Info for '{test_collection_name}': Status: {info.status}, Points: {info.points_count}")

        sample_data = pd.DataFrame([
            {'id': 2000001, 'description': 'Spacious family home with a large garden.', 'price': 600000, 'bedrooms': 4},
            {'id': 2000002, 'description': 'Cozy downtown studio, recently renovated.', 'price': 350000, 'bedrooms': 0},
        ])
        added = db.upsert_points_batch(collection_name=test_collection_name, points_df=sample_data)
        print(f"Test upsert added {added} points.")
        info_after = db.get_collection_info(collection_name=test_collection_name)
        if info_after: print(f"After upsert, points in '{test_collection_name}': {info_after.points_count}")

        test_query = "modern studio apartment"
        results = db.search_points(query_text=test_query, collection_name=test_collection_name, top_k=1)
        if results:
            print(f"\nSearch results for '{test_query}':")
            for hit in results: print(f"  ID={hit.id}, Score={hit.score:.4f}, Desc: {hit.payload.get('description')}")
        else: print(f"No search results for '{test_query}'.")
    else:
        print("\nFailed to initialize VectorDB for testing (client or model missing).")