# Deployment & Setup Guide

## Quick Start

### 1. Installation

```bash
# Clone or navigate to project
cd factual-eval

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate
# Or (Mac/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create `.env` file in project root:
```bash
OPENAI_API_KEY=your-key
ANTHROPIC_API_KEY=your-key
```

Create `config/config.yaml` (optional):
```yaml
model:
  name: claude-haiku-4-5-20251001

rag:
  enabled: true

judges:
  - claude-opus-4-8
  - gpt-4

guardrails:
  enabled: true

risk:
  realtime: true

run:
  default_limit: 20
```

### 3. Run Dashboard

```bash
streamlit run app/app.py
```

Visit `http://localhost:8501` in your browser.

### 4. Run CLI Evaluation

```bash
# Single model, limited questions
python main.py evaluate --models claude-haiku-4-5-20251001 --limit 20

# Multiple models
python main.py evaluate --models claude-haiku-4-5-20251001 gpt-4 --limit 100

# Compare existing results
python main.py compare

# Benchmark quick test
python main.py benchmark
```

### 5. Load Corpus for RAG

```python
from src.rag import load_corpus

# Prepare JSON: [{"text": "..."}, {"text": "..."}, ...]
load_corpus("data/sample_corpus.json")

# Now retrieval will work
from src.rag import retrieve
docs = retrieve("What is the moon?", top_k=5)
```

## Testing

```bash
# Run all tests
pytest -v

# Run specific test file
pytest tests/test_hallucination_classifier.py -v

# Run with coverage
pytest --cov=src tests/
```

## Production Deployment

### Option 1: Heroku

```bash
# Create Procfile
echo "web: streamlit run app/app.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile

# Deploy
git push heroku main
```

### Option 2: Docker + AWS ECS

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
# Build
docker build -t factual-eval .

# Run
docker run -p 8501:8501 -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY factual-eval
```

### Option 3: Docker + Google Cloud Run

```bash
# Build and push
docker build -t gcr.io/YOUR_PROJECT/factual-eval .
docker push gcr.io/YOUR_PROJECT/factual-eval

# Deploy
gcloud run deploy factual-eval \
  --image gcr.io/YOUR_PROJECT/factual-eval \
  --platform managed \
  --region us-central1 \
  --set-env-vars ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY
```

### Option 4: Docker + Azure Container Instances

```bash
# Build and push to ACR
az acr build --registry myregistry --image factual-eval:latest .

# Deploy
az container create \
  --resource-group mygroup \
  --name factual-eval \
  --image myregistry.azurecr.io/factual-eval:latest \
  --environment-variables ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  --ports 8501
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `ANTHROPIC_API_KEY` | Claude API key | Yes |
| `OPENAI_API_KEY` | OpenAI API key | Optional |
| `JUDGE_MODEL` | Default judge model | No (defaults to claude-opus-4-8) |
| `RAG_ENABLED` | Enable RAG retrieval | No (defaults to true) |
| `STREAMLIT_SERVER_PORT` | Port for Streamlit | No (defaults to 8501) |

## Database Setup (for leaderboard at scale)

### Option 1: PostgreSQL

```sql
CREATE TABLE leaderboard (
  id SERIAL PRIMARY KEY,
  timestamp TIMESTAMP DEFAULT NOW(),
  model VARCHAR(255),
  hallucination_rate FLOAT,
  groundedness FLOAT,
  avg_risk_score FLOAT,
  citation_coverage FLOAT,
  num_evaluations INT,
  metadata JSONB
);

CREATE INDEX idx_model ON leaderboard(model);
CREATE INDEX idx_timestamp ON leaderboard(timestamp DESC);
```

### Option 2: Firebase Firestore

```python
from firebase_admin import firestore

db = firestore.client()
db.collection('leaderboard').add({
    'model': 'claude-opus',
    'hallucination_rate': 0.15,
    'timestamp': firestore.SERVER_TIMESTAMP,
})
```

## Scaling Tips

1. **Cache Judge Responses**: Responses are cached in `cache/judge_*.json` to avoid recomputation
2. **Batch Processing**: Use `--limit` to process in chunks
3. **Distributed Evaluation**: Run multiple instances in parallel with different model subsets
4. **Vector DB Optimization**: Use Pinecone/Weaviate instead of FAISS for production scale
5. **Load Balancing**: Put Streamlit app behind nginx/cloud LB for traffic distribution

## Monitoring & Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/eval.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
```

## Troubleshooting

### Issue: FAISS index not building
```
Solution: Ensure you have faiss-cpu installed: pip install faiss-cpu
```

### Issue: Streamlit connection timeout
```
Solution: Check API keys in .env file, verify network connectivity
```

### Issue: Out of memory with large corpus
```
Solution: Use approximate index in FAISS:
from faiss import IndexIVFFlat
index = IndexIVFFlat(dimension, n_clusters)
```

### Issue: Leaderboard CSV getting large
```
Solution: Archive old entries, use database backend for production
```

## Support & Contribution

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Contributing**: See CONTRIBUTING.md
- **Paper**: ArXiv link (coming soon)
- **License**: MIT
