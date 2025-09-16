# Reverse Image Search — FastAPI (BLIP × CLIP)

**One‑file FastAPI backend** that captions an image with **BLIP**, embeds it with **CLIP**, and (optionally) stores/queries vectors in a vector database for reverse image search.

> Current main feature: `/embed-image/` endpoint that returns a BLIP description and a CLIP embedding for an uploaded image. Vector DB persistence/search is scaffolded and can be wired to MongoDB Atlas Vector Search or another store.

---

## Features

- **BLIP captioning** — generate a natural‑language description from an image.
- **CLIP embeddings** — compute image/text embeddings for similarity search.
- **Simple, single‑file API** — FastAPI app you can run with Uvicorn.
- **Vector DB‑ready** — imports for MongoDB are present; wire up Atlas Vector Search (or pgvector, FAISS, Pinecone) as you prefer.

---

## Tech Stack

- **Python** · **FastAPI** · **Uvicorn**
- **Hugging Face Transformers**: BLIP, CLIP
- **PyTorch** for model execution
- **Pillow (PIL)** for image I/O
- **pymongo** (optional) for MongoDB/Atlas Vector Search integration

---

## Quickstart

### 1) Setup

```bash
# (Recommended) create a virtual env
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

# install dependencies
pip install -r requirements.txt
```

> If you have a GPU + CUDA, ensure your PyTorch install matches your CUDA version for best performance (see pytorch.org). CPU works too, just slower.

### 2) Run the API

```bash
uvicorn main:app --reload
```

The server defaults to `http://127.0.0.1:8000`.

- Interactive docs: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

### 3) Try the endpoint

**cURL**
```bash
curl -X POST "http://127.0.0.1:8000/embed-image/" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/image.jpg"
```

**Python**
```python
import requests
files = {"file": open("image.jpg", "rb")}
r = requests.post("http://127.0.0.1:8000/embed-image/", files=files)
print(r.json())
```

Example JSON response:
```json
{
  "description": "a person riding a motorcycle on a city street",
  "embedding_vector": [0.0123, -0.0456, ...]
}
```

---

## API

### `POST /embed-image/`

- **Body (form-data):** `file` — image (jpg/png/etc.)
- **Returns:** JSON with
  - `description` (string): BLIP caption
  - `embedding_vector` (list[float]): CLIP image embedding

> Notes:
> - The main file currently defines helpers `gen_caption(image)` and `gen_embedding(image, caption)` that are used inside this endpoint.
> - A sample `/items/{item_id}` route is also present for basic testing.

---

## Configuration

The code imports `MongoClient` for persistence, but it’s not yet wired. If you want to enable storage + search:

- **MongoDB Atlas Vector Search** (suggested path):
  1. Create a collection with a `vector` field of the correct dimensionality (e.g., 512/768 depending on CLIP model).
  2. Store documents as `{ _id, description, vector, metadata... }`.
  3. Create a vector index on the `vector` field.
  4. On upload:
     - Generate caption + embedding, insert into the collection.
  5. For search:
     - Query using `$vectorSearch` (nearest neighbor) with the query vector.

- **Alternatives:** pgvector (Postgres), FAISS (local), Pinecone, Qdrant, Weaviate, etc.

### Environment variables (suggested)
- `MONGODB_URI` — Atlas or local Mongo connection string
- `MONGODB_DB` — database name
- `MONGODB_COL` — collection name
- `BLIP_MODEL` — e.g., `Salesforce/blip-image-captioning-base`
- `CLIP_MODEL` — e.g., `openai/clip-vit-base-patch32`

> If you stick to defaults in code, you may not need all of these. To switch models, set variables and read them in the initialization section.

---

## Model Notes

- **BLIP**: produces a short image caption. Good defaults:
  - `Salesforce/blip-image-captioning-base` (fast)
  - `Salesforce/blip-image-captioning-large` (better, heavier)
- **CLIP**: produces embeddings that align images and text in the same space. Popular:
  - `openai/clip-vit-base-patch32`
  - `openai/clip-vit-large-patch14`

Hugging Face caches models on first run (typically under `~/.cache/huggingface`).

---

## Project Structure

```
.
├── main.py            # FastAPI app (this file)
├── requirements.txt   # Python deps
└── ...                # (optional) any extra scripts/config
```

---

## Roadmap

- [ ] Add `/index` endpoint to persist `{description, vector, metadata}` to a DB
- [ ] Add `/search` endpoint (k-NN over vectors) returning similar images/IDs
- [ ] Batch/indexing CLI for folders of images
- [ ] Auth + rate limiting for public deployments
- [ ] Dockerfile for reproducible deploys

---

## License

MIT — do whatever you want, but no warranty. See `LICENSE` (add one if missing).

---

## Acknowledgements

- BLIP — https://huggingface.co/docs/transformers/model_doc/blip
- CLIP — https://huggingface.co/docs/transformers/model_doc/clip
- FastAPI — https://fastapi.tiangolo.com/
- MongoDB Atlas Vector Search — https://www.mongodb.com/products/platform/atlas-vector-search (optional)
