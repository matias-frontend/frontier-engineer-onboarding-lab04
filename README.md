# hybrid-search

A hybrid search engine: Pinecone vector search fused with a from-scratch BM25
keyword index, combined by either weighted scoring or Reciprocal Rank Fusion.

Frontier Engineer Onboarding, Day 4 / Lab 04. See
[`docs/DEVIATIONS.md`](docs/DEVIATIONS.md) for where this departs from the
handout — the Pinecone SDK's call shapes have changed, and the handout's
embedding provider cannot be deployed.

## Why hybrid

The two retrievers fail in opposite directions, which is the whole argument for
running both:

| Query | Vector search | BM25 |
| --- | --- | --- |
| `"Okapi"` | weak — a rare proper noun barely moves an embedding | **strong** — exact match, high IDF |
| `"how do I prepare noodles from scratch"` | **strong** — matches the pasta doc with no shared words | weak — no term overlap at all |

Fusion keeps both wins. `scripts/compare.ts` demonstrates this on a fixed query
set.

## Requirements

- Node **20.x** (see `.node-version`)
- `PINECONE_API_KEY` — free Starter plan at [app.pinecone.io](https://app.pinecone.io)
- `GOOGLE_API_KEY` — free at [aistudio.google.com](https://aistudio.google.com)
- `COHERE_API_KEY` — *optional*, enables the rerank extension

> Gemini's free tier allows roughly 20 chat requests/day **per model**, but
> embeddings draw on a **separate** quota — which is most of what this lab uses.
> If you do hit a wall, switching `GEMINI_MODEL` gets you a fresh allowance.

## Setup

```bash
npm install
export PINECONE_API_KEY=... GOOGLE_API_KEY=...
```

## Verify

```bash
npm run smoke:local   # 56 checks: chunking, BM25, fusion — no keys, no network
npm run smoke:e2e     # end-to-end against real Pinecone + Gemini
npm run compare       # RRF vs weighted vs each arm alone, plus a k sweep
```

`smoke:local` is the one to run while iterating — the retrieval maths is where
the logic actually lives, and it's fully testable offline.

## Run it

```bash
npm start
```

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/index` | `{ doc_id, title, text }` — chunk, embed, index into both stores |
| POST | `/search` | `{ query, limit?, vector_weight?, bm25_weight?, method?, k?, filter?, rerank? }` |
| GET | `/health` | Status, BM25 doc count, Pinecone vector count |
| POST | `/rehydrate` | Rebuild the BM25 index from Pinecone |

```bash
curl -X POST http://localhost:8000/index -H "Content-Type: application/json" -d '{
  "doc_id": "doc1",
  "title": "Introduction to Hybrid Search",
  "text": "Hybrid search combines the strengths of vector-based semantic search with traditional keyword-based search..."
}'

curl -X POST http://localhost:8000/search -H "Content-Type: application/json" \
  -d '{"query": "how does semantic search work with keywords", "limit": 5}'

curl -X POST http://localhost:8000/search -H "Content-Type: application/json" \
  -d '{"query": "vector and keyword combination", "limit": 5, "method": "rrf"}'
```

Every result carries `hybrid_score`, `vector_score`, `bm25_score`, and the
`vector_rank` / `bm25_rank` it held in each source list — so you can see *why*
something ranked where it did, not just that it did.

## Deploy to Render

1. Push to GitHub.
2. Render → **New → Blueprint**, point it at the repo (`render.yaml`).
3. Set `PINECONE_API_KEY` and `GOOGLE_API_KEY` when prompted.
4. Verify: `curl https://<your-app>.onrender.com/health`

No disk is required. Pinecone holds the vectors and BM25 rebuilds itself from
Pinecone on boot, so the free tier's ephemeral filesystem costs nothing here.

## Extension challenges implemented

- **RRF vs weighted comparison** — `npm run compare` runs both across
  exact-term, conceptual, and mixed queries and reports where they disagree,
  plus an RRF `k` sweep (20/60/100). `docs/DEVIATIONS.md` §7 records the
  concrete failure mode of weighted scoring that this surfaced.
- **Metadata filtering** — `filter` on `/search`, applied to *both* arms
  (Pinecone server-side, BM25 in-process) so the two can't disagree about
  what's eligible.
- **Cohere Rerank** — `rerank: true` on `/search`, active only when
  `COHERE_API_KEY` is set. A reranker outage degrades the ranking rather than
  failing the search.
- **Contextual retrieval** — not implemented; see DEVIATIONS §6.

## Layout

```
src/
  config.ts       env + tuning (weights, dimensions, chunk size)
  chunking.ts     sentence-aware overlapping chunker
  bm25.ts         BM25 from scratch, with an inverted index
  fusion.ts       weighted + RRF fusion (pure, no I/O — fully unit-testable)
  embeddings.ts   Gemini embeddings at 384 dims
  pinecone.ts     index lifecycle, upsert, query
  hydrate.ts      rebuild BM25 from Pinecone at boot
  search.ts       indexing + search orchestration
  rerank.ts       optional Cohere rerank
  retry.ts        exponential backoff with jitter
  quota.ts        distinguishes transient 429 from spent daily quota
  server.ts       Hono API
scripts/
  smoke-local.ts  chunking + BM25 + fusion (no keys needed)
  smoke-e2e.ts    end-to-end against real services
  compare.ts      strategy comparison + RRF k sweep
```
