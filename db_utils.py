# db_utils.py

import os
import json
import uuid
import time
import numpy as np
from typing import List, Dict, Any
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
import requests

# Load environment variables
load_dotenv()

# Constants
DB_DIR = os.getenv("DB_DIR", "./chromadb")
OLLAMA_API = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
N_CONTEXTS = int(os.getenv("N_CONTEXTS", "3"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"

# Initialize ChromaDB client with telemetry disabled
client = chromadb.PersistentClient(path=DB_DIR, settings=Settings(anonymized_telemetry=False))

# Global variable to store the ChromaDB collection
vector_db = None

def debug_print(message):
    if DEBUG_MODE:
        print(f"DEBUG: {message}")

def ensure_collection_exists():
    global vector_db
    try:
        vector_db = client.get_or_create_collection(name='conversations')
        debug_print(f"Using existing or created new collection 'conversations'. Count: {vector_db.count()}")
    except Exception as e:
        debug_print(f"Error in ensure_collection_exists: {str(e)}")

def get_embedding(prompt: str) -> List[float]:
    try:
        debug_print(f"Getting embedding for prompt: {prompt[:50]}...")  # Print first 50 chars of prompt
        response = requests.post(f"{OLLAMA_API}/api/embeddings", json={"model": EMBED_MODEL, "prompt": prompt})
        response.raise_for_status()
        embedding = response.json().get('embedding')
        if embedding is None:
            debug_print("Embedding is None in API response")
        else:
            debug_print(f"Successfully got embedding of length {len(embedding)}")
        return embedding
    except requests.RequestException as e:
        debug_print(f"Error getting embedding: {str(e)}")
        return None

def add_to_vector_db(conversation: Dict[str, str]):
    global vector_db
    ensure_collection_exists()
    
    try:
        debug_print("Adding new conversation to vector DB")
        embedding = get_embedding(conversation['prompt'] + " " + conversation['response'])
        if embedding is None:
            debug_print("Failed to get embedding, skipping vector DB update")
            return
        
        conversation_id = str(uuid.uuid4())
        conversation['id'] = conversation_id  # Add ID to the conversation dict
        metadata = {"id": conversation_id, "timestamp": time.time()}
        
        vector_db.add(
            ids=[conversation_id],
            embeddings=[embedding],
            documents=[json.dumps(conversation)],
            metadatas=[metadata]
        )
        debug_print(f"Successfully added to vector DB with ID: {conversation_id}")
    except Exception as e:
        debug_print(f"Error in add_to_vector_db: {str(e)}")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_context(prompt: str) -> List[Dict[str, Any]]:
    global vector_db
    ensure_collection_exists()
    
    try:
        debug_print(f"Retrieving context for prompt: {prompt[:50]}...")  # Print first 50 chars of prompt
        prompt_embedding = get_embedding(prompt)
        if prompt_embedding is None:
            debug_print("Failed to get embedding for prompt, returning empty context")
            return []
        
        debug_print(f"Querying vector DB with embedding of length {len(prompt_embedding)}")
        results = vector_db.query(
            query_embeddings=[prompt_embedding],
            n_results=N_CONTEXTS,
            include=['documents', 'embeddings', 'metadatas']
        )
        
        debug_print(f"Raw query results: {json.dumps(results, indent=2)}")
        
        contexts = []
        if isinstance(results, dict) and 'documents' in results and results['documents']:
            documents = results['documents'][0]
            embeddings = results['embeddings'][0]
            metadatas = results.get('metadatas', [[]])[0]
            
            for doc, embedding, metadata in zip(documents, embeddings, metadatas):
                try:
                    context = json.loads(doc)
                    similarity = cosine_similarity(prompt_embedding, embedding)
                    
                    # Handle cases where metadata might be None
                    if metadata is not None:
                        context['id'] = metadata.get('id') or context.get('id', 'Unknown')
                    else:
                        context['id'] = context.get('id', 'Unknown')
                    
                    context['similarity'] = similarity
                    contexts.append(context)
                except json.JSONDecodeError:
                    debug_print(f"Error decoding document: {doc}")
            
            # Sort contexts by similarity (highest first) and take top N_CONTEXTS
            contexts.sort(key=lambda x: x['similarity'], reverse=True)
            contexts = contexts[:N_CONTEXTS]
        
        debug_print(f"Retrieved {len(contexts)} contexts")
        for idx, context in enumerate(contexts, 1):
            debug_print(f"Context {idx} (similarity: {context['similarity']:.4f}):")
            debug_print(f"  ID: {context['id']}")
            debug_print(f"  Prompt: {context['prompt']}")
            debug_print(f"  Response: {context['response'][:50]}...")  # Truncate long responses
        return contexts
    except Exception as e:
        debug_print(f"Error in retrieve_context: {str(e)}")
        debug_print(f"Exception details: {traceback.format_exc()}")
        return []

# Initialize the collection when the module is imported
ensure_collection_exists()