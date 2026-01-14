"""
Word Counter Tool - FastAPI Backend
"""

import os
import re
import uuid
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Word Counter API",
    description="A simple word counter API with Firestore storage",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Firestore initialization
db = None

def get_firestore_client():
    """Initialize and return Firestore client."""
    global db
    if db is not None:
        return db

    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        # Check if already initialized
        try:
            firebase_admin.get_app()
        except ValueError:
            # Initialize Firebase
            cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if cred_path and os.path.exists(cred_path):
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred)
            else:
                # Use default credentials (for Cloud Run, etc.)
                firebase_admin.initialize_app()

        db = firestore.client()
        return db
    except Exception as e:
        print(f"Firestore initialization error: {e}")
        return None


# Pydantic models
class TextInput(BaseModel):
    text: str
    title: Optional[str] = None


class WordCountResult(BaseModel):
    id: str
    title: str
    text: str
    word_count: int
    character_count: int
    character_count_no_spaces: int
    sentence_count: int
    paragraph_count: int
    created_at: str


class WordCountStats(BaseModel):
    word_count: int
    character_count: int
    character_count_no_spaces: int
    sentence_count: int
    paragraph_count: int


class SearchQuery(BaseModel):
    query: str
    search_in: str = "both"  # "title", "content", or "both"


def count_words(text: str) -> WordCountStats:
    """Count words, characters, sentences, and paragraphs in text."""
    # Word count
    words = text.strip().split() if text.strip() else []
    word_count = len(words)

    # Character counts
    character_count = len(text)
    character_count_no_spaces = len(text.replace(" ", "").replace("\t", "").replace("\n", ""))

    # Sentence count (split by .!?)
    sentences = re.split(r'[.!?]+', text)
    sentence_count = len([s for s in sentences if s.strip()])

    # Paragraph count (split by blank lines)
    paragraphs = re.split(r'\n\s*\n', text)
    paragraph_count = len([p for p in paragraphs if p.strip()])

    # Ensure at least 1 paragraph if there's text
    if text.strip() and paragraph_count == 0:
        paragraph_count = 1

    return WordCountStats(
        word_count=word_count,
        character_count=character_count,
        character_count_no_spaces=character_count_no_spaces,
        sentence_count=sentence_count,
        paragraph_count=paragraph_count
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Word Counter API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    firestore_status = "connected" if get_firestore_client() else "disconnected"
    return {"status": "healthy", "firestore": firestore_status}


@app.post("/api/word-count", response_model=WordCountResult)
async def create_word_count(input_data: TextInput):
    """Count words and save to Firestore."""
    text = input_data.text
    title = input_data.title or f"Entry {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"

    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    # Calculate stats
    stats = count_words(text)

    # Generate ID and timestamp
    doc_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat() + "Z"

    # Create result object
    result = WordCountResult(
        id=doc_id,
        title=title,
        text=text,
        word_count=stats.word_count,
        character_count=stats.character_count,
        character_count_no_spaces=stats.character_count_no_spaces,
        sentence_count=stats.sentence_count,
        paragraph_count=stats.paragraph_count,
        created_at=created_at
    )

    # Save to Firestore
    firestore_client = get_firestore_client()
    if firestore_client:
        try:
            doc_ref = firestore_client.collection("word_counts").document(doc_id)
            doc_ref.set(result.model_dump())
        except Exception as e:
            print(f"Firestore write error: {e}")
            # Continue even if Firestore fails - return the result anyway

    return result


@app.get("/api/word-count/history", response_model=list[WordCountResult])
async def get_history(limit: int = 20):
    """Get word count history from Firestore."""
    firestore_client = get_firestore_client()

    if not firestore_client:
        return []

    try:
        docs = (
            firestore_client.collection("word_counts")
            .order_by("created_at", direction="DESCENDING")
            .limit(limit)
            .stream()
        )

        results = []
        for doc in docs:
            data = doc.to_dict()
            # Handle entries without title (backward compatibility)
            if "title" not in data:
                data["title"] = f"Entry {data.get('created_at', '')[:10]}"
            results.append(WordCountResult(**data))

        return results
    except Exception as e:
        print(f"Firestore read error: {e}")
        return []


@app.get("/api/word-count/{doc_id}", response_model=WordCountResult)
async def get_word_count(doc_id: str):
    """Get a specific word count entry."""
    firestore_client = get_firestore_client()

    if not firestore_client:
        raise HTTPException(status_code=503, detail="Database unavailable")

    try:
        doc = firestore_client.collection("word_counts").document(doc_id).get()

        if not doc.exists:
            raise HTTPException(status_code=404, detail="Entry not found")

        data = doc.to_dict()
        # Handle entries without title (backward compatibility)
        if "title" not in data:
            data["title"] = f"Entry {data.get('created_at', '')[:10]}"
        return WordCountResult(**data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/word-count/{doc_id}")
async def delete_word_count(doc_id: str):
    """Delete a word count entry."""
    firestore_client = get_firestore_client()

    if not firestore_client:
        raise HTTPException(status_code=503, detail="Database unavailable")

    try:
        doc_ref = firestore_client.collection("word_counts").document(doc_id)
        doc = doc_ref.get()

        if not doc.exists:
            raise HTTPException(status_code=404, detail="Entry not found")

        doc_ref.delete()
        return {"message": "Entry deleted successfully", "id": doc_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/word-count/analyze", response_model=WordCountStats)
async def analyze_text(input_data: TextInput):
    """Analyze text without saving to database."""
    text = input_data.text
    return count_words(text)


@app.get("/api/word-count/search", response_model=list[WordCountResult])
async def search_entries(q: str, search_in: str = "both", limit: int = 50):
    """Search word count entries by title or content."""
    firestore_client = get_firestore_client()

    if not firestore_client:
        return []

    if not q.strip():
        return []

    query_lower = q.lower().strip()

    try:
        # Firestore doesn't support full-text search natively,
        # so we fetch recent entries and filter client-side
        docs = (
            firestore_client.collection("word_counts")
            .order_by("created_at", direction="DESCENDING")
            .limit(200)  # Fetch more to filter
            .stream()
        )

        results = []
        for doc in docs:
            data = doc.to_dict()

            # Handle entries without title (backward compatibility)
            if "title" not in data:
                data["title"] = f"Entry {data.get('created_at', '')[:10]}"

            title_match = query_lower in data.get("title", "").lower()
            content_match = query_lower in data.get("text", "").lower()

            if search_in == "title" and title_match:
                results.append(WordCountResult(**data))
            elif search_in == "content" and content_match:
                results.append(WordCountResult(**data))
            elif search_in == "both" and (title_match or content_match):
                results.append(WordCountResult(**data))

            if len(results) >= limit:
                break

        return results
    except Exception as e:
        print(f"Search error: {e}")
        return []


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
