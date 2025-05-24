import os
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data') 
RAW_DATA_PATH = os.path.join(DATA_DIR, 'raw', 'kc_house_data.csv')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed', 'kc_house_data_processed.csv')
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333") 
QDRANT_GRPC_PORT = int(os.getenv("QDRANT_GRPC_PORT", "6334")) 
QDRANT_COLLECTION_NAME = "king_county_houses"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" 
VECTOR_SIZE = 384 