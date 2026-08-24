# Deviations from the lab handout

## 1. Pinecone SDK v8 call shapes differ from the handout's hints

The handout says "`upsert()` accepts a list of `{id, values, metadata}` objects".
In `@pinecone-database/pinecone` v8 it does not — the calls are:

| Handout | Actual (v8) |
| --- | --- |
| `index.upsert([...])` | `index.upsert({ records: [...] })` |
| `index.fetch([...ids])` | `index.fetch({ ids: [...] })` |
| `deleteMany({ doc_id })` | `deleteMany({ filter: { doc_id } })` |

Found by typechecking against the installed package rather than trusting the
prose. `createIndex({ name, dimension, metric, spec: { serverless } })` and
`query({ vector, topK, includeMetadata, filter })` are as documented.

Pinecone's current quickstart pushes the *integrated* embedding API
(`createIndexForModel` / `upsertRecords` / `searchRecords`), which embeds
server-side. This lab requires bring-your-own 384-dim vectors, so the classic
API is the correct one here.

## 2. Embeddings: Gemini at 384 dimensions, not Ollama

The handout specifies Ollama for TypeScript. **Ollama cannot run on a deployed
container** — there is no Ollama process beside the app — so the handout's own
"deploy to Railway or Render" deliverable is incompatible with its embedding
choice.

Running the same MiniLM model in-process via transformers.js was evaluated in
Lab 03 and rejected on measurement: 388–642 MB of dependencies carrying
unfixable high/critical CVEs through an unused `sharp` dependency.

Gemini's embedding model accepts an explicit `outputDimensionality`, so setting
it to **384** matches the lab's required Pinecone dimension exactly. The whole
project stays at ~82 MB with a clean `npm audit`, and embeddings bill against a
separate quota from chat completions.

## 3. `gemini-2.0-flash` is shut down

Step 5 names it for contextual retrieval. It no longer exists, and
`gemini-2.5-flash` is closed to new API keys. `GEMINI_MODEL` defaults to
`gemini-3.6-flash`.

## 4. BM25 is rebuilt from Pinecone at startup

**This is the most substantive change.** The handout keeps BM25 purely in
memory next to a persistent Pinecone index. On any restart the vectors survive
and the keyword index does not, so hybrid search silently degrades to
vector-only while still reporting itself as hybrid, and `/health` shows
`bm25_docs: 0`. On a free-tier host that spins down when idle, that is the
normal state, not an edge case.

Since Pinecone metadata already stores each chunk's text, the vector store is a
sufficient source of truth. `src/hydrate.ts` pages through the index at boot and
refills BM25. This keeps **one** authoritative store rather than introducing a
second one to keep in sync, and it is why `render.yaml` needs no disk.

`POST /rehydrate` exposes the same operation on demand.

## 5. Additions beyond the handout

- **Bounded-concurrency embedding.** Firing a 50-chunk document at the API in
  parallel reliably trips the free-tier per-minute limit; the retry backoff then
  serialises them anyway, slower and noisier. `embedBatch` paces the requests.
- **Filters applied to both arms.** Pinecone filters server-side; BM25 is
  in-process, so the same predicate is applied there too. Without that the
  filter would appear to work while keyword hits leaked past it.
- **Index dimension is checked at startup.** A mismatch between `EMBEDDING_DIM`
  and an existing index is reported as an actionable error rather than surfacing
  later as a confusing upsert failure. An index's dimension is immutable.
- **Health check survives a dead Pinecone.** `/health` answers with
  `pinecone_vectors: null` rather than failing, and the server starts serving
  before bootstrap completes — otherwise the platform kills the deploy during a
  slow index warm-up.
- **Retry and quota handling** carried over from Lab 03: 503 and per-minute 429
  get exponential backoff with jitter; an exhausted daily quota fails fast,
  because it will not clear for hours.

## 6. Contextual retrieval (Step 5) not implemented

Step 5 is marked optional in the handout and is also extension challenge 3. It
is out of scope for this build and is genuinely absent — not stubbed.

## 7. Expanded stop-word list

The handout specifies exactly 21 stop words. That list is too short for
natural-language queries, and the comparison run proved it: the query
*"how do I prepare noodles from scratch"* retrieved a **TF-IDF document**,
because that document contains "how" twice and "how" carried real IDF weight
against a corpus where it was otherwise rare.

Question words and auxiliary verbs are precisely the tokens a conversational
query is built from *and* precisely the ones carrying no topical signal, so
leaving them in makes BM25 actively noisy on the query type it is worst at
already. `STOP_WORDS` in `src/bm25.ts` keeps the handout's list and adds
interrogatives, pronouns, and high-frequency verbs.

After the change the same query tokenises to `["prepare","noodles","scratch"]`
and BM25 correctly returns nothing for it — leaving the semantic arm to answer,
which is the right division of labour.

## 8. Measured: hybrid search can make conceptual queries *worse*

The most useful result from `npm run compare`, and a caveat the handout does
not mention.

For the query *"why does vector length not affect the score"*, the correct
document is `cmp_cosine` — the one stating that cosine similarity "ignores
their magnitude, which makes it scale invariant". Vector search ranked it
**first**. Both fusion methods pushed it **out of the top three**:

```
vector-only : cmp_cosine, cmp_embeddings, cmp_tfidf     <- correct
weighted    : cmp_tfidf, cmp_embeddings, cmp_bm25       <- correct answer lost
rrf         : cmp_bm25, cmp_embeddings, cmp_tfidf       <- correct answer lost
```

BM25 matched "length", "score", and "vector" lexically in documents about
*length normalisation* and *scoring* — terms that are topically adjacent but
answer a different question. Those spurious matches outweighed the one genuinely
correct semantic hit.

The conclusion is not that hybrid retrieval is bad. It is that **fusion weights
are a per-query-type decision, not a constant**. A query with no rare
identifiers has nothing for BM25 to contribute and everything for it to pollute.
A production system would classify the query first — or, more cheaply, apply a
cross-encoder rerank over the fused set, which is what `src/rerank.ts` exists
for.

This is also the honest answer to the lab's "document which approach handles
which query types best":

| Query type | Best strategy | Why |
| --- | --- | --- |
| Rare exact term (`Okapi`, `autolyse`) | BM25, or hybrid | High IDF makes it unmissable; embeddings blur rare tokens |
| Pure paraphrase, common vocabulary | **Vector alone** | BM25 contributes noise and can displace the right answer |
| Mixed (term + concept) | Hybrid | Both arms agree, and fusion reinforces the overlap |

## 9. What the test suite proved about weighted vs RRF

Worth recording, since extension challenge 1 asks for exactly this comparison.

Min-max normalisation maps the **worst** item in a list to exactly 0. So a chunk
that ranks last among vector hits and first among BM25 hits contributes only its
BM25 share — appearing in *both* lists does **not** guarantee a higher weighted
score than a chunk that merely topped one list.

`scripts/smoke-local.ts` pins this with concrete numbers (0.3 vs 0.7 on a fixed
input) and then shows RRF promoting the both-list item to first on the same
data, because RRF reads rank rather than raw score. That is the practical
argument for RRF when you do not control the score distributions.
