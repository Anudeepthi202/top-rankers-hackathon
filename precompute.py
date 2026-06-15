# precompute.py
import json
import numpy as np
from sentence_transformers import SentenceTransformer

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load JD and compute its embedding
with open("job_description.md", "r", encoding="utf-8") as f:
    jd_text = f.read()
jd_emb = model.encode(jd_text, normalize_embeddings=True)
np.save("jd_embedding.npy", jd_emb)

# Load all candidates (uncompressed .jsonl)
candidates = []
with open("candidates.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            candidates.append(json.loads(line))
print(f"Loaded {len(candidates)} candidates")

# Prepare texts for embedding (summary + latest job description)
candidate_ids = []
texts = []
for c in candidates:
    candidate_ids.append(c["candidate_id"])
    summary = c["profile"].get("summary", "")
    latest_job = ""
    if c["career_history"]:
        latest_job = c["career_history"][0].get("description", "")
    text = (summary + " " + latest_job).strip()
    if not text:
        text = " "
    texts.append(text)

print("Encoding candidates...")
embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

np.save("candidate_embeddings.npy", embeddings)
np.save("candidate_ids_embedding.npy", np.array(candidate_ids))
print("Pre-computation done.")