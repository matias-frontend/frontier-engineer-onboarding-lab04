/**
 * Search API.
 *
 *   POST /index    index a document into Pinecone + BM25
 *   POST /search   hybrid search (weighted or RRF), optional filter + rerank
 *   GET  /health   liveness, with BM25 document count
 *   POST /rehydrate  rebuild BM25 from Pinecone on demand
 */
import { serve } from '@hono/node-server';
import { Hono } from 'hono';

import { bm25 } from './bm25.js';
import { BM25_WEIGHT, HOST, PORT, REHYDRATE_ON_BOOT, RRF_K, VECTOR_WEIGHT } from './config.js';
import { rehydrateBm25 } from './hydrate.js';
import { ensureIndex, indexStats } from './pinecone.js';
import { isDailyQuotaExhausted, quotaGuidance } from './quota.js';
import { rerankEnabled } from './rerank.js';
import { indexDocument, search, type SearchMethod } from './search.js';

const app = new Hono();

function errorResponse(error: unknown): { body: Record<string, unknown>; status: 429 | 500 } {
    const detail = error instanceof Error ? error.message : String(error);
    if (isDailyQuotaExhausted(error)) {
        return {
            body: { error: 'Gemini free-tier daily quota exhausted.', detail: quotaGuidance(error), upstream: detail },
            status: 429
        };
    }
    return { body: { error: detail }, status: 500 };
}

app.get('/health', async c => {
    const stats = await indexStats().catch(() => null);
    return c.json({
        status: 'ok',
        bm25_docs: bm25.totalDocs,
        pinecone_vectors: stats?.vectors ?? null,
        pinecone_dimension: stats?.dimension ?? null,
        rerank_available: rerankEnabled()
    });
});

app.post('/index', async c => {
    const body = (await c.req.json().catch(() => null)) as Record<string, unknown> | null;
    if (!body) {
        return c.json({ error: 'Request body must be a JSON object.' }, 400);
    }

    const { doc_id: docId, title, text } = body;
    if (typeof docId !== 'string' || !docId.trim()) {
        return c.json({ error: 'Field "doc_id" is required and must be a non-empty string.' }, 400);
    }
    if (typeof title !== 'string' || !title.trim()) {
        return c.json({ error: 'Field "title" is required and must be a non-empty string.' }, 400);
    }
    if (typeof text !== 'string' || !text.trim()) {
        return c.json({ error: 'Field "text" is required and must be a non-empty string.' }, 400);
    }

    try {
        return c.json(await indexDocument(docId.trim(), title.trim(), text));
    } catch (error) {
        const { body: payload, status } = errorResponse(error);
        console.error('[server] /index failed:', payload.error);
        return c.json(payload, status);
    }
});

app.post('/search', async c => {
    const body = (await c.req.json().catch(() => null)) as Record<string, unknown> | null;
    if (!body) {
        return c.json({ error: 'Request body must be a JSON object.' }, 400);
    }

    const query = body.query;
    if (typeof query !== 'string' || !query.trim()) {
        return c.json({ error: 'Field "query" is required and must be a non-empty string.' }, 400);
    }

    const method = (body.method ?? 'weighted') as SearchMethod;
    if (method !== 'weighted' && method !== 'rrf') {
        return c.json({ error: 'Field "method" must be either "weighted" or "rrf".' }, 400);
    }

    const limit = body.limit === undefined ? 10 : Number(body.limit);
    if (!Number.isInteger(limit) || limit < 1 || limit > 100) {
        return c.json({ error: 'Field "limit" must be an integer between 1 and 100.' }, 400);
    }

    const vectorWeight = body.vector_weight === undefined ? VECTOR_WEIGHT : Number(body.vector_weight);
    const bm25Weight = body.bm25_weight === undefined ? BM25_WEIGHT : Number(body.bm25_weight);
    for (const [name, value] of [['vector_weight', vectorWeight], ['bm25_weight', bm25Weight]] as const) {
        if (!Number.isFinite(value) || value < 0 || value > 1) {
            return c.json({ error: `Field "${name}" must be a number between 0 and 1.` }, 400);
        }
    }

    const filter = body.filter;
    if (filter !== undefined && (typeof filter !== 'object' || filter === null || Array.isArray(filter))) {
        return c.json({ error: 'Field "filter", when provided, must be an object.' }, 400);
    }

    try {
        const response = await search(query.trim(), {
            limit,
            vectorWeight,
            bm25Weight,
            method,
            k: body.k === undefined ? RRF_K : Number(body.k),
            filter: filter as Record<string, unknown> | undefined,
            rerank: body.rerank === true
        });
        return c.json(response);
    } catch (error) {
        const { body: payload, status } = errorResponse(error);
        console.error('[server] /search failed:', payload.error);
        return c.json(payload, status);
    }
});

app.post('/rehydrate', async c => {
    try {
        return c.json(await rehydrateBm25());
    } catch (error) {
        const { body: payload, status } = errorResponse(error);
        return c.json(payload, status);
    }
});

async function bootstrap(): Promise<void> {
    await ensureIndex();
    if (REHYDRATE_ON_BOOT) {
        const result = await rehydrateBm25();
        console.log(
            `[hybrid-search] BM25 rehydrated: ${result.documents} chunks in ${result.durationMs}ms` +
                (result.truncated ? ' (truncated at the cap)' : '')
        );
    }
}

// Serve immediately, bootstrap in the background: the health check must be
// answerable while Pinecone is still warming up, or the platform kills the
// deploy before it ever becomes ready.
serve({ fetch: app.fetch, port: PORT, hostname: HOST }, info => {
    console.log(`[hybrid-search] listening on http://${HOST}:${info.port}`);
});

bootstrap().catch(error => {
    console.error('[hybrid-search] bootstrap failed:', error instanceof Error ? error.message : error);
});

export { app };
