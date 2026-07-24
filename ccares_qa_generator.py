import os
import json
import pandas as pd
from tqdm import tqdm
from groq import Groq
from dotenv import load_dotenv
from docx import Document

# -------------------------------
# Load API Key
# -------------------------------
load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

MODEL = "llama-3.1-8b-instant"

# -------------------------------
# Read DOCX
# -------------------------------

def read_docx(path):

    doc = Document(path)

    text = []

    for para in doc.paragraphs:
        if para.text.strip():
            text.append(para.text.strip())

    return "\n".join(text)


# -------------------------------
# Chunk Text
# -------------------------------

def chunk_text(text, chunk_size=2500, overlap=300):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunks.append(text[start:end])

        start += chunk_size - overlap

    return chunks


# -------------------------------
# Generate QA
# -------------------------------

def generate_qa(context):

    prompt = f"""
You are creating a supervised fine-tuning dataset.

Read the context carefully.

Generate EXACTLY 40 Question-Answer pairs.

Rules:

- Questions should be diverse.
- Include factual questions.
- Include procedural questions.
- Include role-based questions.
- Do NOT hallucinate.
- Answers must ONLY come from the context.
- Return ONLY valid JSON.

Format:

[
    {{
        "question":"...",
        "answer":"..."
    }}
]

Context:

{context}

"""

    response = client.chat.completions.create(

        model=MODEL,

        temperature=0.4,

        response_format={"type":"json_object"},

        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ]
    )

    text = response.choices[0].message.content

    return text


# -------------------------------
# Main
# -------------------------------

def main():

    input_path = r"C:\\Geetika\\Office-Fine Tuning\\Officers_Manual_2.0_(3).docx"

    output_path = "synthetic_dataset__.csv"

    print("Reading DOCX...")

    text = read_docx(input_path)

    print("Chunking...")

    chunks = chunk_text(text)

    print("Total chunks:", len(chunks))

    rows = []

    for chunk in tqdm(chunks):

        try:

            output = generate_qa(chunk)

            data = json.loads(output)

            if isinstance(data, dict):
                data = data["pairs"]

            for qa in data:

                rows.append({
                    "question": qa["question"],
                    "answer": qa["answer"]
                })

        except Exception as e:

            print(e)

            continue

    df = pd.DataFrame(rows)

    df.to_csv(output_path,index=False)

    print(f"\nGenerated {len(df)} QA pairs.")

    print("Saved to:",output_path)


if __name__ == "__main__":
    main()