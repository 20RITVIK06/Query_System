# Document Query Assistant

An intelligent AI assistant for querying insurance, legal, HR, and compliance documents using semantic search and structured reasoning.

## Architecture

- **FastAPI**: API gateway handling requests
- **Pinecone**: Vector database for semantic clause retrieval
- **Google Gemini**: Semantic reasoning and structured response generation
- **PostgreSQL**: Document metadata and interaction logs
- **Redis**: Response caching for speed optimization

## Features

- Semantic document clause retrieval
- Intelligent query analysis with condition extraction
- Structured JSON responses
- Response caching for performance
- Configurable scoring thresholds

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and database URLs
   ```

3. **Run the API:**
   ```bash
   python main.py
   ```

## API Usage

### Query Endpoint

**POST** `/query`

```json
{
  "user_question": "What are the coverage limits for medical expenses?",
  "top_k_clauses": [
    {
      "clause_id": "policy_001",
      "content": "Medical expenses covered up to $50,000...",
      "score": 0.95
    }
  ]
}
```

**Response:**
```json
{
  "answer": "Medical expenses are covered up to $50,000 per incident",
  "conditions": ["treatment within 30 days", "licensed healthcare provider"],
  "clause_id": "policy_001",
  "score": 0.95,
  "explanation": "This clause directly addresses medical expense limits"
}
```

### Health Check

**GET** `/health`

Returns service status.

## Testing

Run the example test:
```bash
python test_example.py
```

## Configuration

Key environment variables:
- `GEMINI_API_KEY`: Your Google Gemini API key
- `GEMINI_MODEL`: Gemini model to use (default: gemini-pro)
- `PINECONE_API_KEY`: Your Pinecone API key
- `REDIS_URL`: Redis connection URL
- `DATABASE_URL`: PostgreSQL connection URL
- `MIN_SCORE_THRESHOLD`: Minimum relevance score (default: 0.7)

## Response Format

All query responses follow this structure:
- `answer`: Direct answer to the user's question
- `conditions`: List of requirements, limitations, or exclusions
- `clause_id`: ID of the most relevant clause used
- `score`: Semantic relevance score (0-1)
- `explanation`: Brief reasoning for the response