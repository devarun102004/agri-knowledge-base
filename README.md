# Pesticide Knowledge Base PoC

This project follows the PoC flow: extract PDF pages and detected tables, clean
and validate the extraction, write a Markdown document with YAML front matter,
create source-aware chunks and multilingual embeddings, store them in Qdrant, and
return source-grounded recommendation evidence.

## 1. Setup

Use the existing virtual environment or create one:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Set Qdrant Cloud credentials in `.env`, using `.env.example` as a template:

```dotenv
QDRANT_URL=https://your-cluster-endpoint:6333
QDRANT_API_KEY=your-database-api-key
```

Do not commit `.env` or send its API key in chat. The multilingual FastEmbed
model is downloaded locally on first use (~220 MB); document chunks are sent to
the configured Qdrant collection.

## 2. Add and ingest documents

Place PDF, UTF-8 text, or Markdown files in `datas/raw_pdfs/`, then run:

```bash
python -m scripts.ingest
```

All PDF pages are processed for text and detected tables. Cleaned OKF Markdown
with YAML front matter is written to `datas/okf/`. Chunks, embeddings, source,
page, and clearly labelled crop/disease/product table fields are saved to the
`pesticide_knowledge` Qdrant collection. Re-ingestion updates old chunks for a
source.

You can also pass a file or directory explicitly:

```bash
python -m scripts.ingest ./some-folder --collection pesticide_knowledge
python -m scripts.ingest datas/okf/bio_insecticides_recommendations.md
```

Markdown recommendation records are indexed one record per chunk. Their crop,
disease/insect, product, dosage, application instructions, habitat, and surface
fields are retained as Qdrant metadata when present.

Scanned/image-only PDFs require OCR before ingestion. Table detection depends
on the PDF's layout; check the generated Markdown under `datas/okf/` before
relying on a table-derived recommendation.

## 3. Retrieve a structured, source-only result

```bash
python -m scripts.search "maize fall armyworm"
```

The default JSON response includes crop, disease/pest, product, active
ingredient, evidence passage, source file, page, and relevance score. Product
and ingredient are populated only when they were found under recognizable table
headers; otherwise the evidence passage is returned without guessing.

Verify any product against its current local registration and label. This PoC
retrieves evidence; it does not prescribe application rates or replace
professional/local guidance.

## 4. End-to-end checks

```bash
python -m unittest discover -s tests -v
python -m compileall -q config src scripts tests
```
