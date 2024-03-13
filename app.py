from fastapi import FastAPI
from embedding import get_embedding_model, generate_embeddings

# Fast API
app = FastAPI()

# Reading model
embedding_model = get_embedding_model()

@app.get("/")
async def root():
    return {"message": "Embedding model is online."}

@app.get('/embedding')
async def embed(prompt: str):
    print(prompt)
    embeddings = generate_embeddings(embedding_model, prompt)
    return {'prompt': prompt, 'embedding': str(embeddings)}

if __name__ == "__main__":
    app.run()
