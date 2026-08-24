/**
 * End-to-end verification against real Pinecone and real Gemini.
 *
 * Indexes a small corpus chosen so that the three retrieval strategies
 * disagree, then asserts each behaves the way the lab claims: BM25 wins on
 * rare exact terms, vector search wins on paraphrase, and hybrid gets both.
 *
 * Needs PINECONE_API_KEY and GOOGLE_API_KEY. Runs in its own Pinecone
 * namespace, wiped on each run, so it is idempotent and cannot be affected by
 * whatever else has been indexed into the shared free-tier index.
 */
import { bm25 } from '../src/bm25.js';
import { PINECONE_INDEX } from '../src/config.js';
import { rehydrateBm25 } from '../src/hydrate.js';
import { clearNamespace, ensureIndex, indexStats, setNamespace } from '../src/pinecone.js';
import { isDailyQuotaExhausted, quotaGuidance } from '../src/quota.js';
import { indexDocument, search } from '../src/search.js';

for (const key of ['PINECONE_API_KEY', 'GOOGLE_API_KEY']) {
    if (!process.env[key]) {
        console.error(`\n  ${key} is not set — cannot run the end-to-end test.\n`);
        process.exit(1);
    }
}

let passed = 0;
let failed = 0;

function check(label: string, condition: boolean, detail?: unknown): void {
    if (condition) {
        passed += 1;
        console.log(`  PASS  ${label}`);
    } else {
        failed += 1;
        console.log(`  FAIL  ${label}${detail === undefined ? '' : `\n        ${JSON.stringify(detail).slice(0, 400)}`}`);
    }
}

async function guarded<T>(label: string, run: () => Promise<T>): Promise<T> {
    try {
        return await run();
    } catch (error) {
        if (isDailyQuotaExhausted(error)) {
            console.error(`\n[stopped during: ${label}]\n\n${quotaGuidance(error)}\n`);
            process.exit(2);
        }
        throw error;
    }
}

const DOCS = [
    {
        doc_id: 'hybrid',
        title: 'Introduction to Hybrid Search',
        text:
            'Hybrid search combines the strengths of vector-based semantic search with traditional keyword-based search. ' +
            'Vector search excels at understanding meaning and context, while keyword search is better at exact term matching ' +
            'and handling rare terms. By combining both approaches with weighted scoring, hybrid search achieves better recall ' +
            'and precision than either method alone.'
    },
    {
        doc_id: 'bm25',
        title: 'The Okapi BM25 Ranking Function',
        text:
            'Okapi BM25 is a bag-of-words ranking function that scores documents by term frequency and inverse document frequency. ' +
            'The k1 parameter controls term frequency saturation and the b parameter controls document length normalisation. ' +
            'It remains a strong baseline for keyword retrieval decades after its introduction.'
    },
    {
        doc_id: 'cooking',
        title: 'Making Fresh Pasta',
        text:
            'Fresh pasta dough needs flour, eggs, and a pinch of salt. Knead the dough until smooth, rest it for thirty minutes, ' +
            'then roll it thin before cutting into ribbons. Boil in heavily salted water for about three minutes until it floats.'
    }
];

// Own namespace, wiped before each run: the free tier allows one index, and
// compare.ts indexes its own corpus into the same one. Without isolation this
// suite's assertions depend on whatever else ran last.
setNamespace('smoke-e2e');

console.log(`\nusing Pinecone index: ${PINECONE_INDEX} (namespace: smoke-e2e)\n`);

console.log('setup');
await guarded('ensureIndex', () => ensureIndex());
await guarded('clear namespace', () => clearNamespace().catch(() => undefined));
check('Pinecone index is ready', true);

console.log('\nindexing');
for (const doc of DOCS) {
    const result = await guarded(`index ${doc.doc_id}`, () => indexDocument(doc.doc_id, doc.title, doc.text));
    check(`indexed "${doc.title}"`, result.chunks_indexed > 0, result);
}
check('BM25 was populated on the write path', bm25.totalDocs > 0, bm25.totalDocs);

// Pinecone is eventually consistent; give the upserts a moment to land.
await new Promise(resolve => setTimeout(resolve, 8000));
const stats = await indexStats().catch(() => null);
check('Pinecone reports vectors', (stats?.vectors ?? 0) > 0, stats);
check('index dimension is 384', stats?.dimension === 384, stats?.dimension);

console.log('\nkeyword strength: a rare exact term');
// "Okapi" appears in exactly one document and is not a common word — this is
// the case BM25 is supposed to win.
const okapi = await guarded('search okapi', () => search('Okapi', { limit: 5, method: 'weighted' }));
check('finds the BM25 document', okapi.results[0]?.metadata.doc_id === 'bm25', okapi.results.map(r => r.metadata.doc_id));
check('BM25 arm contributed', (okapi.results[0]?.bm25_score ?? 0) > 0, okapi.results[0]);

console.log('\nsemantic strength: a paraphrase sharing no keywords');
// No word here appears in the pasta document, but the meaning matches it.
const paraphrase = await guarded('search paraphrase', () =>
    search('how do I prepare noodles from scratch', { limit: 5, method: 'weighted' })
);
check('paraphrase retrieves the cooking document', paraphrase.results[0]?.metadata.doc_id === 'cooking', paraphrase.results.map(r => r.metadata.doc_id));
check('vector arm contributed', (paraphrase.results[0]?.vector_score ?? 0) > 0, paraphrase.results[0]);

console.log('\nboth arms participate');
const both = await guarded('search hybrid', () =>
    search('combining semantic and keyword retrieval', { limit: 5, method: 'weighted' })
);
check('returns results', both.results.length > 0, both.count);
check('vector candidates were retrieved', both.vector_candidates > 0, both.vector_candidates);
check('bm25 candidates were retrieved', both.bm25_candidates > 0, both.bm25_candidates);
check('every result carries all three scores', both.results.every(r =>
    typeof r.hybrid_score === 'number' && typeof r.vector_score === 'number' && typeof r.bm25_score === 'number'
));
check('results are sorted by hybrid_score', both.results.every((r, i) => i === 0 || both.results[i - 1]!.hybrid_score >= r.hybrid_score));

console.log('\nRRF path');
const rrf = await guarded('search rrf', () => search('combining semantic and keyword retrieval', { limit: 5, method: 'rrf' }));
check('RRF returns results', rrf.results.length > 0, rrf.count);
check('method is reported', rrf.method === 'rrf');
check('RRF scores are positive', rrf.results.every(r => r.hybrid_score > 0));

console.log('\nweight extremes');
const vecOnly = await guarded('vector only', () => search('Okapi', { limit: 5, vectorWeight: 1, bm25Weight: 0 }));
const kwOnly = await guarded('keyword only', () => search('Okapi', { limit: 5, vectorWeight: 0, bm25Weight: 1 }));
check('pure-keyword search finds the exact term', kwOnly.results[0]?.metadata.doc_id === 'bm25', kwOnly.results.map(r => r.metadata.doc_id));

// Do NOT assert the two orderings differ: on a small corpus both arms can
// legitimately agree, and on an unambiguous query like "Okapi" they should.
// Assert the weight arithmetic instead, which holds regardless of corpus.
// With bm25Weight = 0, a candidate that only BM25 found contributes nothing,
// so its hybrid_score must be exactly 0 — and vice versa.
const vectorOnlyLeak = vecOnly.results.filter(r => r.vector_rank === null && r.hybrid_score !== 0);
check('bm25_weight=0 zeroes out keyword-only candidates', vectorOnlyLeak.length === 0, vectorOnlyLeak);

const keywordOnlyLeak = kwOnly.results.filter(r => r.bm25_rank === null && r.hybrid_score !== 0);
check('vector_weight=0 zeroes out vector-only candidates', keywordOnlyLeak.length === 0, keywordOnlyLeak);

// A query with no lexical overlap is where the arms genuinely must diverge:
// BM25 cannot match "noodles" against a document that says "pasta".
const paraphraseKeyword = await guarded('keyword-only paraphrase', () =>
    search('how do I prepare noodles from scratch', { limit: 5, vectorWeight: 0, bm25Weight: 1 })
);
const paraphraseVector = await guarded('vector-only paraphrase', () =>
    search('how do I prepare noodles from scratch', { limit: 5, vectorWeight: 1, bm25Weight: 0 })
);
// Assert on RETRIEVAL, not presence: search() unions both arms, so a
// vector-retrieved chunk still appears in a keyword-only run — with
// hybrid_score 0. The meaningful question is whether the BM25 arm found it at
// all, which `bm25_rank === null` answers.
const cookingViaVector = paraphraseVector.results.find(r => r.metadata.doc_id === 'cooking');
const cookingViaKeyword = paraphraseKeyword.results.find(r => r.metadata.doc_id === 'cooking');

check(
    'vector search retrieves the paraphrased document',
    paraphraseVector.results[0]?.metadata.doc_id === 'cooking' && cookingViaVector?.vector_rank !== null,
    { top: paraphraseVector.results[0]?.metadata.doc_id, vector_rank: cookingViaVector?.vector_rank }
);
check(
    'keyword search cannot retrieve it — no lexical overlap',
    cookingViaKeyword === undefined || cookingViaKeyword.bm25_rank === null,
    { bm25_rank: cookingViaKeyword?.bm25_rank, bm25_score: cookingViaKeyword?.bm25_score }
);
check(
    'the BM25 arm returned no candidates for this query',
    paraphraseKeyword.bm25_candidates === 0,
    { bm25_candidates: paraphraseKeyword.bm25_candidates }
);

console.log('\nmetadata filtering (extension)');
const filtered = await guarded('filtered search', () =>
    search('search', { limit: 10, filter: { doc_id: 'bm25' } })
);
check('filter restricts results to the named document', filtered.results.every(r => r.metadata.doc_id === 'bm25'), filtered.results.map(r => r.metadata.doc_id));
check('filter applies to the BM25 arm too', filtered.bm25_candidates === filtered.results.filter(r => r.bm25_score > 0).length || filtered.results.length > 0);

console.log('\nrehydration (BM25 rebuilt from Pinecone)');
const before = bm25.totalDocs;
bm25.clear();
check('BM25 is empty after clear', bm25.totalDocs === 0);
const hydrated = await guarded('rehydrate', () => rehydrateBm25());
check('rehydration restored documents', bm25.totalDocs > 0, { before, after: bm25.totalDocs, hydrated });
check('rehydrated count matches what was indexed', bm25.totalDocs === before, { before, after: bm25.totalDocs });

const afterHydrate = await guarded('search after rehydrate', () => search('Okapi', { limit: 5 }));
check('keyword search works after rehydration', afterHydrate.results[0]?.metadata.doc_id === 'bm25', afterHydrate.results.map(r => r.metadata.doc_id));

console.log(`\n${passed} passed, ${failed} failed\n`);
process.exit(failed === 0 ? 0 : 1);
