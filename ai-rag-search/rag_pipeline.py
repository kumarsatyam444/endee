import os

import openai

from .search import semantic_search

openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key:
    openai.api_key = openai_api_key


def construct_prompt(question: str, retrieved_docs: list) -> str:
    context_chunks = []
    for i, doc in enumerate(retrieved_docs, start=1):
        context_chunks.append(f"{i}. {doc.get('text','').strip()}")

    context = "\n\n".join(context_chunks)
    return (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer clearly and concisely."
    )


def generate_answer(question: str, top_k: int = 5) -> str:
    if not question or not question.strip():
        return "Please provide a non-empty question."

    retrieved = semantic_search(question, top_k=top_k)
    if len(retrieved) == 0:
        return "No relevant context found for your question."

    prompt = construct_prompt(question, retrieved)

    if not openai_api_key:
        # Fallback local generation for demo mode when API key is absent
        summary = "\n\n".join([doc['text'] for doc in retrieved])
        return (
            "[OpenAI key not set. Returning context summary.]\n\n"
            f"Question: {question}\n\n"
            f"Relevant context:\n{summary[:1700]}..."
        )

    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions from retrieved documents."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=400,
    )

    answer = resp["choices"][0]["message"]["content"].strip()
    return answer
