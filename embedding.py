from sentence_transformers import SentenceTransformer
import os
import config

def get_embedding_model():
    """
    Generator llm read using langchain from hugging face hub
    
    """
    if os.path.isdir(config.model_path):
        model = SentenceTransformer(config.model_path)
    else:
        model = SentenceTransformer(config.model_name)
        model.save(config.model_path)
    return model

def generate_embeddings(model, prompt):
    embeddings = model.encode(prompt)
    return embeddings