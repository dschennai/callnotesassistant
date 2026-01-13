# Word Counter Tool

A full-stack word counter application with text analysis and history tracking.

## Tech Stack

- **Frontend**: Next.js 14 (App Router) + React 18 + Tailwind CSS
- **Backend**: FastAPI (Python) on Port 8000
- **Database**: Google Firestore (NoSQL)

## Features

- Real-time word, character, sentence, and paragraph counting
- Save text entries to Firestore for history tracking
- View and manage saved entries
- Load previous entries for editing
- RESTful API for text analysis

## Project Structure

```
callnotesassistant/
├── frontend/                 # Next.js frontend
│   ├── src/
│   │   ├── app/             # App Router pages
│   │   │   ├── layout.tsx
│   │   │   ├── page.tsx
│   │   │   └── globals.css
│   │   └── components/
│   │       └── WordCounter.tsx
│   ├── package.json
│   ├── next.config.js
│   ├── tailwind.config.js
│   └── tsconfig.json
├── backend/                  # FastAPI backend
│   ├── main.py
│   ├── requirements.txt
│   └── .env.example
└── WORD_COUNTER_README.md
```

## Setup Instructions

### Prerequisites

- Node.js 18+ and npm
- Python 3.9+
- Google Cloud account with Firestore enabled

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up Firestore credentials:
   - Create a Firebase/Firestore project at https://console.firebase.google.com
   - Go to Project Settings > Service Accounts
   - Generate a new private key (JSON file)
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Update `GOOGLE_APPLICATION_CREDENTIALS` with the path to your JSON file

5. Run the backend server:
   ```bash
   uvicorn main:app --reload --port 8000
   ```

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Run the development server:
   ```bash
   npm run dev
   ```

4. Open http://localhost:3000 in your browser

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check with Firestore status |
| POST | `/api/word-count` | Count words and save to Firestore |
| GET | `/api/word-count/history` | Get saved word count history |
| GET | `/api/word-count/{id}` | Get specific entry |
| DELETE | `/api/word-count/{id}` | Delete an entry |
| POST | `/api/word-count/analyze` | Analyze text without saving |

### Example API Usage

**Count and save text:**
```bash
curl -X POST http://localhost:8000/api/word-count \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world. This is a test."}'
```

**Get history:**
```bash
curl http://localhost:8000/api/word-count/history
```

## Firestore Data Structure

Collection: `word_counts`

Document structure:
```json
{
  "id": "uuid-string",
  "text": "The original text",
  "word_count": 3,
  "character_count": 18,
  "character_count_no_spaces": 15,
  "sentence_count": 1,
  "paragraph_count": 1,
  "created_at": "2024-01-15T10:30:00.000Z"
}
```

## Development

### Running Both Services

For development, run both services in separate terminals:

**Terminal 1 - Backend:**
```bash
cd backend
source venv/bin/activate
uvicorn main:app --reload --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

The frontend proxies API requests to the backend via `next.config.js` rewrites.
