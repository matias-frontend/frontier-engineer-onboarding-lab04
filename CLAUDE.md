# hybrid-search

Pinecone vector search + from-scratch BM25, fused by weighted scoring or RRF.
Hono API.

## Commands

```bash
npm run smoke:local   # chunking + BM25 + fusion; no keys, no network, fast
npm run smoke:e2e     # end-to-end against real Pinecone + Gemini
npm run compare       # RRF vs weighted across query types + k sweep
npm run build         # typecheck only
npm start             # serve on :8000
```

Run `smoke:local` after touching `chunking.ts`, `bm25.ts`, or `fusion.ts` — it
covers the actual logic and costs nothing.

## Invariants

- **`fusion.ts` imports no I/O.** It is pure ranking maths, which is what makes
  56 assertions runnable with no API keys. Keep Pinecone/Gemini/BM25 out of it.
- **Fusion unions, never intersects.** A chunk only vector search found, and one
  only BM25 found, are exactly the cases hybrid search exists to catch.
- **Filters must apply to both arms.** Pinecone filters server-side; BM25 is
  in-process and needs the same predicate applied in `search.ts`. Skipping it
  makes the filter look like it works while keyword hits leak through.
- **`EMBEDDING_DIM` must equal the Pinecone index dimension.** An index's
  dimension is immutable — changing one means recreating the other. `ensureIndex`
  checks this at startup and fails with an actionable message.
- **BM25 rebuilds from Pinecone, not from a second store.** One source of truth.
  Don't add a local database to persist it.

## Gotchas

- Pinecone SDK v8 shapes differ from the lab handout: `upsert({ records })`,
  `fetch({ ids })`, `deleteMany({ filter })`. Typecheck rather than trusting docs.
- Pinecone is eventually consistent — `smoke:e2e` waits after upserting before
  asserting on counts. Don't remove that pause.
- `GEMINI_MODEL` defaults to `gemini-3.6-flash`. Never `gemini-2.0-flash` (shut
  down) or `gemini-2.5-flash` (404s for new keys).
- Embeddings and chat draw on **separate** Gemini quotas; this project is mostly
  embeddings, which is the more generous bucket.
- Don't add transformers.js for local embeddings — evaluated and rejected in
  Lab 03 (400–640MB, unfixable CVEs via `sharp`).
- `/health` must keep answering when Pinecone is unreachable. A health check that
  dies with its dependency gets the deploy killed before it can report anything.
