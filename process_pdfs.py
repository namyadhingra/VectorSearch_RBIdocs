# This script processes legal and regulatory PDFs, extracts text, chunks it, computes embeddings

# Run these to install dependencies in the virtual environment:
# pip install numpy pdfplumber pdf2image pillow pytesseract torch transformers sentence-transformers
# pip install pymilvus
# tesseract and poppler must be installed in system and added to PATH

# tip: if running in VS code, of course first enure environment is activated
# and also, choose "Run Python File" in VS Code

import os
import shutil
import glob
import re
import random
import numpy as np
import pdfplumber
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

# Model being used for dense embeddings: Alibaba-NLP/gte-large-en-v1.5
# Model being used for sparse embeddings: Bag of Words (BOW) with max 5000 words

# ---------- TEXT EXTRACTION AND CHUNKING ----------
def extract_text_pdfplumber(pdf_path):
    text = ''
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + '\n'
        if text.strip() == '':
            return None
        return text
    except Exception as e:
        print(f"pdfplumber error: {e}")
        return None

def extract_text_ocr(pdf_path, temp_dir="./_temp_images"):
    os.makedirs(temp_dir, exist_ok=True)
    text = ''
    pages = convert_from_path(pdf_path, dpi=300, output_folder=temp_dir, fmt='png', thread_count=4)
    for i, page in enumerate(pages):
        image_path = os.path.join(temp_dir, f"page_{i}.png")
        page.save(image_path)
        img = Image.open(image_path)
        page_text = pytesseract.image_to_string(img, lang='eng')
        text += page_text + '\n'
        img.close()
    shutil.rmtree(temp_dir)
    return text

"""
Paragraph Chunking seems to be the best chunking method for our use case for the following reasons:
1. Legal and regulatory documents are formatted with meaningful paragraphs, each usually focused on a single rule or topic.
2. Paragraph chunking keeps citations, clauses, and context together, which is vital for answer retrieval, compliance checks, and semantic embeddings
"""
# Previous chunking methods like sentence or fixed-size chunks were not as effective:
    # paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    # paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

def chunk_by_paragraph(
    text,
    min_length=200,
    max_length=1200,
    hard_max_length=15000  # WELL UNDER MILVUS LIMIT TO BE SAFE
    # Milvus limit: 16384 chars
):
    import re
    lines = text.splitlines()
    paragraphs = []
    current = ""

    for line in lines:
        line = line.strip()
        is_section_header = (
            bool(re.match(r"^\d+\.", line)) or
            bool(re.match(r"^\([a-zA-Z]\)", line)) or
            line.isupper()
        )
        if is_section_header and current.strip():
            paragraphs.append(current.strip())
            current = line
        else:
            if len(line) == 0:
                if current.strip():
                    paragraphs.append(current.strip())
                    current = ""
            else:
                current += " " + line

    if current.strip():
        paragraphs.append(current.strip())

    # --- Merge, but never allow ANY chunk above hard_max_length ---
    chunks = []
    tmp = ""
    for para in paragraphs:
        # Force-clip very long paragraphs
        if len(para) > hard_max_length:
            if tmp.strip() and len(tmp.strip()) >= min_length:
                chunks.append(tmp.strip())
                tmp = ""
            for i in range(0, len(para), hard_max_length):
                sub = para[i:i+hard_max_length]
                if len(sub) >= min_length:
                    chunks.append(sub)
            continue

        # Merge paras until limit reached
        if (
            len(tmp) + len(para) + 1 <= max_length
            and len(tmp) + len(para) + 1 <= hard_max_length
        ):
            tmp += " " + para
        else:
            if len(tmp.strip()) >= min_length:
                chunks.append(tmp.strip())
            tmp = para

    for ch in chunks:
        assert len(ch) <= hard_max_length, f"WAARRNNIIINGG: Chunk exceeds hard max: {len(ch)}"


    if tmp.strip():
        for i in range(0, len(tmp.strip()), hard_max_length):
            sub = tmp.strip()[i:i+hard_max_length]
            if len(sub) >= min_length:
                chunks.append(sub)

    return chunks


# ---------- SPARSE VECTOR (BOW) ----------
def build_vocab(texts, max_vocab=5000):
    from collections import Counter
    counter = Counter()
    for text in texts:
        words = re.findall(r'\w+', text.lower())
        counter.update(words)
    most_common = counter.most_common(max_vocab)
    vocab = {word: idx for idx, (word, _) in enumerate(most_common)}
    return vocab

def sparse_vector(text, vocab):
    vec = np.zeros(len(vocab), dtype=np.float32)
    words = re.findall(r'\w+', text.lower())
    for w in words:
        idx = vocab.get(w, -1)
        if idx != -1:
            vec[idx] += 1.
    # L2 normalize (Milvus needs normalized vecs)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec

# ---------- DENSE VECTOR (USING GTE-LARGE) ----------
def average_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    return (last_hidden * mask).sum(1) / mask.sum(1)

class GTEEmbedder:
    def __init__(self, model_name="Alibaba-NLP/gte-large-en-v1.5"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()
    def encode(self, texts):
        # input-> list of strings
        with torch.no_grad():
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=8192,
                return_tensors='pt'
            )
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            outputs = self.model(**inputs)
            emb = average_pool(outputs.last_hidden_state, inputs['attention_mask'])
            emb = F.normalize(emb, p=2, dim=1)
            return emb.cpu().numpy()

# ---------- CONNECTING WITH MILVUS ETC. ----------
def connect_milvus(host="localhost", port="19535"):
    connections.connect("default", host=host, port=port)
def create_collection(name, dense_dim, sparse_dim):
    if utility.has_collection(name):
        utility.drop_collection(name)
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="dense", dtype=DataType.FLOAT_VECTOR, dim=dense_dim),
        FieldSchema(name="sparse", dtype=DataType.FLOAT_VECTOR, dim=sparse_dim),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=16384)
    ]
    schema = CollectionSchema(fields, description="RBI Document Collection")
    collection = Collection(name, schema)
    collection.create_index("dense", {"index_type":"IVF_FLAT", "metric_type":"L2", "params":{"nlist":128}})
    collection.create_index("sparse", {"index_type":"IVF_FLAT", "metric_type":"L2", "params":{"nlist":128}})
    collection.load()
    return collection

def insert_to_milvus(collection, dense_vecs, sparse_vecs, texts):
    # autogen id
    entities = [
        dense_vecs.tolist(),
        sparse_vecs.tolist(),
        texts,
    ]
    collection.insert(entities)
    collection.flush()

# ---------- DEFINING SEARCH FUNCTIONS FOR DENSE, SPARSE, AND HYBRID SEARCH ----------
def dense_search(collection, embedder, query, top_k=10):
    dv = embedder.encode([query])
    results = collection.search(dv.tolist(), "dense", {"metric_type":"L2", "params":{"nprobe":10}}, limit=top_k)
    top = [(hit.id, hit.score, hit.entity.get("text")) for hit in results[0]]
    return top

def sparse_search(collection, vocab, query, top_k=10):
    sv = sparse_vector(query, vocab).reshape(1, -1)
    results = collection.search(sv.tolist(), "sparse", {"metric_type":"L2", "params":{"nprobe":10}}, limit=top_k)
    top = [(hit.id, hit.score, hit.entity.get("text")) for hit in results[0]]
    return top

def hybrid_search(collection, embedder, vocab, query, top_k=10):
    dv = embedder.encode([query])
    sv = sparse_vector(query, vocab).reshape(1, -1)
    res_d = collection.search(dv.tolist(), "dense", {"metric_type":"L2", "params":{"nprobe":10}}, limit=top_k*2)
    res_s = collection.search(sv.tolist(), "sparse", {"metric_type":"L2", "params":{"nprobe":10}}, limit=top_k*2)
    # merge by avg rank
    scores = {}
    for i, hit in enumerate(res_d[0]):
        scores[hit.id] = [hit.score, None, hit.entity.get("text")]
    for i, hit in enumerate(res_s[0]):
        if hit.id in scores:
            scores[hit.id][1] = hit.score
        else:
            scores[hit.id] = [None, hit.score, hit.entity.get("text")]
    # Combine: score = average of distances (lower is better in L2)
    hybrid = []
    for id_, (ds, ss, text) in scores.items():
        ds = ds if ds is not None else 10e6
        ss = ss if ss is not None else 10e6
        hybrid.append( (id_, (ds + ss) / 2, text) )
    hybrid.sort(key=lambda x: x[1])
    return hybrid[:top_k]

# ---------- EVALUATION OF RESULTS ----------
# this will print results to console and also save to output file
# it will run dense, sparse, and hybrid searches for each query
def run_evaluation(collection, embedder, vocab, queries, top_k=10, output_file="output.txt"):
    def log(msg=""):
        print(msg)
        f.write(msg + "\n")

    with open(output_file, "a", encoding="utf-8") as f:
        log(f"\n{'='*15} EVALUATION {'='*15}\n")
        log(f"Total queries: {len(queries)}, top_k={top_k}")

        for qid, query in enumerate(queries, 1):
            log(f"\n[Query {qid}] {query[:120].replace('\n', ' ')} ...")

            log("[Dense]")
            res_d = dense_search(collection, embedder, query, top_k)
            for i, (id_, score, text) in enumerate(res_d, 1):
                log(f"{i:02d} [ID:{id_}] Score:{score:.4f}  |  {text[:80].replace('\n',' ')} ...")

            log("[Sparse]")
            res_s = sparse_search(collection, vocab, query, top_k)
            for i, (id_, score, text) in enumerate(res_s, 1):
                log(f"{i:02d} [ID:{id_}] Score:{score:.4f}  |  {text[:80].replace('\n',' ')} ...")

            log("[Hybrid]")
            res_h = hybrid_search(collection, embedder, vocab, query, top_k)
            for i, (id_, score, text) in enumerate(res_h, 1):
                log(f"{i:02d} [ID:{id_}] Score:{score:.4f}  |  {text[:80].replace('\n',' ')} ...")


# ---------- MAIN PIPELINE FOR OUR USE CASE ----------

# batch insert

    

def main():
    PDF_FOLDER = "RBIdocs"
    CHUNKS_DIR = "chunked_texts"

    print("CUDA available:", torch.cuda.is_available())

    os.makedirs(CHUNKS_DIR, exist_ok=True)
    print("Scanning and extracting PDFs...")
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith('.pdf')]
    all_chunks = []

    # ---- EXTRACT & CHUNK ----
    for f in pdf_files:
        full_path = os.path.join(PDF_FOLDER, f)
        text = extract_text_pdfplumber(full_path)
        if not text:
            print(f"  [{f}] No text found, using OCR fallback!")
            text = extract_text_ocr(full_path)
        # Use the robust chunker (with hard_max_length for Milvus safety)
        chunks = chunk_by_paragraph(text)
        if chunks:
            with open(os.path.join(CHUNKS_DIR, f[:-4]+"_chunks.txt"), "w", encoding="utf-8") as out:
                for ch in chunks:
                    out.write(ch+"\n\n")
            all_chunks += chunks
        print(f"  [{f}] Chunks: {len(chunks)}")

    print(f"Total text chunks: {len(all_chunks)}")
    if not all_chunks:
        print("No document chunks extracted! Exiting.")
        return

    # ---- BOW VOCAB ----
    print("Building sparse vocabulary (BOW)...")
    vocab = build_vocab(all_chunks, max_vocab=5000)
    print(f"Vocabulary size: {len(vocab)}")

    # ---- EMBEDDER ----
    print("Loading GTE-Large model (can take a while)...")
    embedder = GTEEmbedder(model_name="Alibaba-NLP/gte-large-en-v1.5")

    # ---- MILVUS ----
    print("Connecting to Milvus...")
    connect_milvus()
    dense_dim = 1024
    sparse_dim = len(vocab)
    collection_name = "rbi_docs"
    print(f"Creating Milvus collection: {collection_name}")
    collection = create_collection(collection_name, dense_dim, sparse_dim)

    print("Computing embeddings and inserting into Milvus (in batches)...")

    # BATCH = 32
    # changed BATCH from 32 to 8 for test, faster computation
    """
    Using Per-batch splitting:
        When a chunk is longer than MAX_TEXT_LEN, split it into multiple sub-chunks, each ≤ 16384 chars.
        Only send to Milvus those chunks (the new, possibly-split sub-chunks) and generate corresponding dense/sparse vectors for each one.
    """

    BATCH = 8

    for start in range(0, len(all_chunks), BATCH):
        chunk_batch = all_chunks[start:start+BATCH]
        # All chunks are safe (≤ hard_max_length), so direct encoding and insertion
        dense_vecs = embedder.encode(chunk_batch)
        sparse_vecs = np.array([sparse_vector(ch, vocab) for ch in chunk_batch])
        insert_to_milvus(collection, dense_vecs, sparse_vecs, chunk_batch)
        print(f"  Inserted chunk {min(start+BATCH, len(all_chunks))} / {len(all_chunks)}")

    print("All data inserted in Milvus!")

    # ---- EVALUATION ----
    print("Preparing queries for evaluation (100 queries each for dense, sparse-word, hybrid):")
    dense_queries = random.sample(all_chunks, min(100, len(all_chunks)))
    vocab_list = list(vocab.keys())
    sparse_queries = random.sample(vocab_list, min(100, len(vocab_list)))
    hybrid_queries = random.sample(all_chunks, min(50, len(all_chunks))) + random.sample(vocab_list, min(50, len(vocab_list)))
    print("Running evaluation (this may take time)...")

    print("\n=== Dense queries ===")
    run_evaluation(collection, embedder, vocab, dense_queries, top_k=10, output_file="output.txt")

    print("\n=== Sparse queries ===")
    run_evaluation(collection, embedder, vocab, sparse_queries, top_k=10, output_file="output.txt")

    print("\n=== Hybrid queries ===")
    run_evaluation(collection, embedder, vocab, hybrid_queries, top_k=10, output_file="output.txt")
    print("Evaluation complete! Results saved to output.txt")

if __name__ == "__main__":
    main()
