# Architecture & System Design

## Platform Overview

At its validated core this is an evaluation harness for TruthfulQA-style factual
accuracy (LLM-as-judge + semantic baseline; hallucination rate with confidence
intervals; judge validation via Cohen's kappa). Around that core it adds a
**roadmap** of safety/risk layers — RAG groundedness, a weighted risk engine,
real-time stream monitoring, adaptive guardrails, and a dashboard — which are
implemented and unit-tested but not yet validated against human judgments. The
diagram below describes the full intended platform; treat the non-core layers as
in-progress, not as measured results.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        User Interface Layer                          │
│  ┌──────────────┬──────────────┬──────────────┬──────────────────┐ │
│  │  Streamlit   │  Live Eval   │  Upload &    │  Leaderboard &   │ │
│  │  Dashboard   │  Interface   │  Analyze     │  Benchmarks      │ │
│  └──────────────┴──────────────┴──────────────┴──────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    Evaluation Pipeline Layer                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Multi-Judge                Hallucination              RAG  │   │
│  │  Verification               Classifier            Retriever │   │
│  └─────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Groundedness           Citation              Refusal       │   │
│  │  Scorer                 Verification          Quality Scorer │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    Safety & Risk Layer                               │
│  ┌──────────────┬──────────────┬──────────────┬──────────────────┐ │
│  │  Risk Engine │  Topic       │  Adaptive    │  Intervention    │ │
│  │  (Weighted   │  Detection   │  Guardrails  │  Recommender     │ │
│  │   Scoring)   │  (Embeddings)│  (Prompts)   │  (Actions)       │ │
│  └──────────────┴──────────────┴──────────────┴──────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    Research & Analysis Layer                         │
│  ┌──────────────┬──────────────┬──────────────────────────────────┐ │
│  │  Adversarial │  Long-Context│  Statistical Analysis &         │ │
│  │  Testing     │  Evaluation  │  Leaderboard Ranking            │ │
│  └──────────────┴──────────────┴──────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    Data Layer                                        │
│  ┌──────────────┬──────────────┬──────────────┬──────────────────┐ │
│  │  Results CSV │  Cache       │  FAISS Index │  Leaderboard    │ │
│  │              │  (Judges)    │  (Corpus)    │  CSV            │ │
│  └──────────────┴──────────────┴──────────────┴──────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Breakdown

### Core (src/core/)
- **schema.py**: `EvaluationResult` - unified data structure for all evaluations

### Evaluators (src/evaluators/)
1. **hallucination.py**: Semantic & lexical hallucination detection with severity/confidence/type
2. **groundedness.py**: Citation support verification against retrieved documents
3. **refusal_quality.py**: Quality scoring for refusals (helpfulness, educational value)

### RAG Layer (src/rag/)
- **retriever.py**: FAISS-based vector retrieval from document corpus

### Risk & Safety (src/risk/, src/guardrails/)
- **risk/__init__.py**: Weighted risk formula (40% hallucination, 30% unsupported claims, 20% confidence, 10% contradiction)
- **guardrails/manager.py**: Basic guardrail triggering logic
- **guardrails/adaptive.py**: Topic-aware guardrail prompts (strict/moderate/lenient)
- **guardrails/interventions.py**: Risk-triggered interventions (warn, force citations, strict mode, clarification, switch model)

### Utilities (src/utils/)
- **risk_markers.py**: Token-level risk detection (uncertainty, risky phrases, contradictions, citation gaps)
- **topic_classifier.py**: Hybrid topic detection (keyword + embedding-based)

### Research (src/research/)
- **adversarial.py**: Red-team testing for prompt injection, misleading framing, authority manipulation, jailbreaks
- **long_context.py**: Memory drift detection, context loss measurement, entity consistency tracking
- **leaderboard.py**: Model benchmarking and public rankings with composite scoring

### Pipeline (src/pipeline.py)
Orchestrates: Generation → Multi-judge scoring → Hallucination classification → Citation verification → Risk assessment → Topic detection → Adaptive guardrails → Intervention recommendation

### Frontend (app/app.py)
Streamlit dashboard with:
- Live evaluation interface
- Real-time risk visualization
- Model leaderboard
- Document upload & analysis
- Configuration management

## Data Flow

```
Question
   ↓
[Detect Topic] → Determine Guardrail Level
   ↓
[Generate Answer] (with topic-aware system prompt)
   ↓
[Multi-Judge Verification] → Get majority verdict & agreement score
   ↓
[Hallucination Classification] → Severity, Confidence, Type
   ↓
[RAG Retrieval] → Get supporting documents
   ↓
[Citation Verification] → Score groundedness & supported claims
   ↓
[Refusal Quality Scoring] → If applicable
   ↓
[Risk Marker Scanning] → Uncertainty, risky phrases, contradictions
   ↓
[Weighted Risk Scoring] → Combine all factors
   ↓
[Adversarial Testing] → Vulnerability detection
   ↓
[Recommendation Engine] → Suggest interventions
   ↓
[Long-Context Analysis] → Track memory drift, context loss
   ↓
[Leaderboard Update] → Update model rankings
   ↓
[EvaluationResult] → Store and display
```

## Key Technologies

- **LLMs**: Claude (judges), GPT-4 (alternative judges)
- **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2)
- **Vector Store**: FAISS (in-memory, efficient retrieval)
- **Evaluation Framework**: Custom schema + multi-metric aggregation
- **Frontend**: Streamlit (rapid UI development)
- **Visualization**: Plotly (interactive charts)
- **Storage**: CSV (leaderboard), JSON (cache)

## Scalability Considerations

1. **FAISS**: Can handle millions of documents with approximate search
2. **Judge Caching**: Avoids redundant API calls for identical Q&A pairs
3. **Batch Processing**: Process multiple models in parallel
4. **Horizontal Scaling**: Stateless pipeline can run on multiple workers

## Deployment Options

### Local Development
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/app.py
```

### Docker
```dockerfile
FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app/app.py"]
```

### Cloud (AWS/GCP/Azure)
- Deploy Streamlit on App Platform / Cloud Run / Azure App Service
- Use managed FAISS in cloud storage or Pinecone/Weaviate
- Store leaderboard in managed database (RDS/Firestore/Cosmos)

## Future Enhancements

1. **Streaming Token Analysis**: Real-time risk detection during generation
2. **Fine-tuned Classifiers**: Custom hallucination models for specific domains
3. **Multi-modal Evaluation**: Image-based hallucination detection
4. **Federated Evaluation**: Privacy-preserving benchmarking
5. **Active Learning**: Automatically improve weak areas
6. **API Endpoint**: REST/GraphQL interface for integration
