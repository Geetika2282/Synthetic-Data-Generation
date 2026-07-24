import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from groq import Groq
from docx import Document

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# CONFIG
# ----------------------------

DOCX_PATH = r"C:\\Geetika\\Office-Fine Tuning4\\Officers_Manual_2.0_(3).docx"
CSV_PATH = "synthetic_dataset__.csv"
OUTPUT_CSV = "synthetic_dataset_evaluated.csv"

MODEL = "llama-3.1-8b-instant"

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# ----------------------------
# Read DOCX
# ----------------------------

def read_docx(path):

    doc = Document(path)

    text = []

    for para in doc.paragraphs:
        if para.text.strip():
            text.append(para.text.strip())

    return "\n".join(text)


# ----------------------------
# Chunking
# ----------------------------

def chunk_text(text, chunk_size=2500, overlap=300):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunks.append(text[start:end])

        start += chunk_size - overlap

    return chunks


# ----------------------------
# Build embeddings
# ----------------------------

print("Reading document...")

text = read_docx(DOCX_PATH)

chunks = chunk_text(text)

print("Embedding chunks...")

chunk_embeddings = embedding_model.encode(
    chunks,
    normalize_embeddings=True
)


# ----------------------------
# Retrieve best chunk
# ----------------------------

def retrieve_context(question):

    q_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True
    )

    scores = cosine_similarity(
        [q_embedding],
        chunk_embeddings
    )[0]

    idx = np.argmax(scores)

    return chunks[idx], float(scores[idx])


# ----------------------------
# LLM Judge
# ----------------------------

def evaluate(question, answer, context):

    prompt = f"""
You are evaluating a synthetic QA pair.

Determine whether the answer is fully supported by the context.

Context:

{context}

Question:

{question}

Answer:

{answer}

Return ONLY JSON.

{{
    "faithfulness":0-1,
    "correctness":0-1,
    "supported":"YES or NO",
    "reason":"one sentence"
}}
"""

    response = client.chat.completions.create(

        model=MODEL,

        temperature=0,

        response_format={"type":"json_object"},

        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ]
    )

    return json.loads(
        response.choices[0].message.content
    )


# ----------------------------
# Main Evaluation
# ----------------------------

df = pd.read_csv(CSV_PATH)

results = []

for _, row in tqdm(df.iterrows(), total=len(df)):

    context, similarity = retrieve_context(
        row["question"]
    )

    score = evaluate(
        row["question"],
        row["answer"],
        context
    )

    results.append({

        "question": row["question"],

        "answer": row["answer"],

        "similarity": round(similarity,3),

        "faithfulness": score["faithfulness"],

        "correctness": score["correctness"],

        "supported": score["supported"],

        "reason": score["reason"]

    })

out = pd.DataFrame(results)

out.to_csv(
    OUTPUT_CSV,
    index=False
)

print()

print("Evaluation complete.")

print(out.head())

print()

print("Supported:",
      (out.supported=="YES").sum(),
      "/",
      len(out))