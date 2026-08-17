from pathlib import Path
import hashlib
import os

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path

from test_questions import test_cases, answer_test_cases

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY was not found in the .env file.")

openai_client = OpenAI(api_key=api_key)
chroma_client = chromadb.PersistentClient(
    path="chroma_db"
)

def load_document(file_path: str) -> str:
    """Read a text file and return its contents."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Could not find: {file_path}")

    return path.read_text(encoding="utf-8")

def create_document_hash(text: str) -> str:
    """Create a SHA-256 fingerprint for the document."""

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

def chunk_document(text: str) -> list[str]:
    """"Split the document into complete procedure sections."""

    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    chunks = []
    current_chunk = []

    for paragraph in paragraphs:
        if paragraph.endswith("Procedure") and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []

        current_chunk.append(paragraph)

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks

def create_embeddings(chunks: list[str]) -> list[list[float]]:
    """Convert text chunks into numerical embeddings."""

    try:
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=chunks,
        )

    except Exception as error:
        raise RuntimeError(
            f"Failed to create embeddings: {error}"
        ) from error

    return [item.embedding for item in response.data]

def prepare_collection(
        chunks: list[str],
        document_hash: str,
) -> chromadb.Collection:
    """Reuse stored embeddings unless the source document changed."""

    collection = chroma_client.get_or_create_collection(
        name="operations_manual"
    )

    stored_hash = None

    if collection.metadata:
        stored_hash = collection.metadata.get("document_hash")

    if (
        collection.count() > 0
        and stored_hash == document_hash
    ):
        print(
            f"Loaded {collection.count()} exisiting chunks "
            "from Chroma."
        )
        return collection

    if collection.count() > 0:
        print("Document changed. Rebuilding embeddings...")

        chroma_client.delete_collection(
            name="operations_manual"
        )

        collection = chroma_client.create_collection(
            name="operations_manual"
        )
    else:
        print("No stored chunks found. Creating embeddings...")

    embeddings = create_embeddings(chunks)

    chunk_ids = [
        f"chunk-{index}"
        for index in range(len(chunks))
    ]

    metadatas = []

    for index, chunk in enumerate(chunks):
        first_line = chunk.splitlines()[0].strip()

        metadatas.append(
            {
                "source_file": "operations_manual.txt",
                "section_title": first_line,
                "chunk_index": index,
            }
        )

    collection.upsert(
        ids=chunk_ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    collection.modify(
        metadata={
            "document_hash": document_hash
        }
    )

    print(f"Stored {collection.count()} chunks.")

    return collection

def search_chunks(
        question: str,
        collection: chromadb.Collection,
        number_of_results: int = 2,
) -> list [str]:
    """Find the most relevant chunks with their source IDs."""

    question_response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input = question,
    )

    question_embedding = question_response.data[0].embedding

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=number_of_results,
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0]
    ids = results["ids"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not distances or distances[0] > 1.0:
        return []

    if not documents:
        return[]

    retrieved_chunks = []

    for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
        retrieved_chunks.append(
            {
                "id": chunk_id,
                "text": document,
                "source_file": metadata["source_file"],
                "section_title": metadata["section_title"],
                "distance": distance,
            }
        )

    return retrieved_chunks

def run_retrieval_tests(
        collection: chromadb.Collection,
) -> None:
    """Run test questions and check whether retrieval finds the expected topic."""

    print("\nRunning retrieval tests...")

    passed_tests = 0
    total_tests = len(test_cases)

    for test in test_cases:
        question = test["question"]
        expected_topic = test["expected_topic"]

        retrieved_chunks = search_chunks(
            question,
            collection,
        )

        print(f"Question: {question}")

        if expected_topic is None:
            print("Expected: No matching topic")
        else:
            print(f"Expected: {expected_topic}")

        if retrieved_chunks:
            actual_topic = retrieved_chunks[0]["section_title"]
            actual_distance = retrieved_chunks[0]["distance"]

            print(f"Retrieved: {actual_topic}")
            print(f"Distance: {actual_distance}")
        else:
            actual_topic = None
            print("Retrieved: Nothing")

        if actual_topic == expected_topic:
            print("Result: PASS")
            passed_tests += 1
        else:
            print("Result: FAIL")

        print()

    print(
        f"Retrieval Test Summary: "
        f"{passed_tests}/{total_tests} tests passed"
    )

def run_answer_tests(
        collection: chromadb.Collection,
) -> None:
    """"Test whether generated answers include expected key information."""

    print("\nRunning answer-quality tests...\n")

    passed_tests = 0
    total_tests = len(answer_test_cases)

    for test in answer_test_cases:
        question = test["question"]
        expected_keywords = test["expected_keywords"]

        relevant_chunks = search_chunks(
            question,
            collection,
        )

        if not relevant_chunks:
            print(f"Question: {question}")
            print("Result: FAIL - no relevant chunks found\n")
            continue

        answer = generate_answer(
            question,
            relevant_chunks,
        )

        answer_lower = answer.lower()

        missing_keywords = [
            keyword 
            for keyword in expected_keywords
            if keyword.lower() not in answer_lower
        ]

        print(f"Question: {question}")

        if not missing_keywords:
            print("Result: PASS")
            passed_tests += 1
        else:
            print(f"Result: FAIL")
            print(f"Missing keywords: {missing_keywords}")

        print()

    print(
        f"Answer Test Summary: "
        f"{passed_tests}/{total_tests} tests passed"
    )

def generate_answer(
        question: str,
        relevant_chunks: list[dict],
) -> str:
    """Generate an answer using retrieved chunks and include sources."""

    context_sections = []

    for chunk in relevant_chunks:
        context_sections.append(
            f"""
            Source: {chunk['source_file']} - {chunk['section_title']}
            {chunk['text']}
            """.strip()
        )

    context = "\n\n".join(context_sections)

    prompt = f""""
Answer the user's question using only the context below.

Include the supporting source IDs at the end of the answer.

At the end of your answer, list the supporting sources using this format:

Sources:
- operations_manual.txt — Server Overheating Procedure

If the answer is not contained in the context, say:
"I could not find that information in the operation manual."

Context:
{context}

Question:
{question}
"""
    try:
        response = openai_client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
        )

    except Exception as error:
        raise RuntimeError(
            f"Failed to generate answer: {error}"
        ) from error

    return response.output_text

def main() -> None:

    try:
        document = load_document(
            "data/operations_manual.txt"
        )

    except FileNotFoundError as error:
        print(f"File error: {error}")
        raise SystemExit

    document_hash = create_document_hash(document)

    document_chunks = chunk_document(document)

    operations_collection = prepare_collection(
        document_chunks,
        document_hash,
    )

    run_retrieval_tests(
        operations_collection,
    )

    run_answer_tests(
        operations_collection,
    )

    question = input("Ask a question about the operations manual: "
    ).strip()

    if not question:
        print("Please enter a question.")
        raise SystemExit

    try:
        relevant_chunks = search_chunks(
            question,
            operations_collection,
        )

        if not relevant_chunks:
            print(
                "I could not find any relevant information "
                "in the operations manual."
            )
            raise SystemExit

        answer = generate_answer(
            question,
            relevant_chunks,
        )

    except RuntimeError as error:
        print(f"\nApplication error: {error}")
        raise SystemExit

    print("\nAnswer: ")
    print(answer)

if __name__ == "__main__":
    main()