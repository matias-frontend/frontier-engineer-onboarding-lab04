/**
 * Extension challenge 1: RRF vs weighted, and hybrid vs each arm alone.
 *
 * Two things this script has to get right, both of which are easy to get
 * wrong:
 *
 *  1. TRUE ARM ISOLATION. Calling search() with vector_weight=0 does NOT give
 *     you keyword-only results — search() unions both retrievers and a zero
 *     weight merely zeroes the score, leaving the other arm's candidates in
 *     the list and its ordering visible through a stable sort. The single-arm
 *     rows below therefore call the retrievers directly.
 *
 *  2. A CORPUS THAT CAN DISAGREE. With three wildly different documents every
 *     strategy returns the same ranking and the comparison proves nothing. The
 *     corpus here is deliberately confusable: overlapping vocabulary, several
 *     documents per topic, and rare exact identifiers that appear in exactly
 *     one place.
 *
 * Needs PINECONE_API_KEY and GOOGLE_API_KEY.
 */
import { bm25 } from '../src/bm25.js';
import { embedQuery } from '../src/embeddings.js';
import { rrfFusion, weightedFusion, type ScoredCandidate } from '../src/fusion.js';
import { rehydrateBm25 } from '../src/hydrate.js';
import { ensureIndex, queryVectors } from '../src/pinecone.js';
import { isDailyQuotaExhausted, quotaGuidance } from '../src/quota.js';
import { indexDocument } from '../src/search.js';

for (const key of ['PINECONE_API_KEY', 'GOOGLE_API_KEY']) {
    if (!process.env[key]) {
        console.error(`\n  ${key} is not set.\n`);
        process.exit(1);
    }
}

async function guarded<T>(run: () => Promise<T>): Promise<T> {
    try {
        return await run();
    } catch (error) {
        if (isDailyQuotaExhausted(error)) {
            console.error(`\n${quotaGuidance(error)}\n`);
            process.exit(2);
        }
        throw error;
    }
}

/**
 * Confusable corpus: multiple documents per topic with shared vocabulary, so
 * ranking is a real decision rather than a formality.
 */
const CORPUS = [
    {
        doc_id: 'cmp_bm25',
        title: 'BM25 Scoring',
        text:
            'BM25 scores documents using term frequency and inverse document frequency. The k1 parameter controls ' +
            'term frequency saturation so a term repeated many times does not dominate. The b parameter controls ' +
            'document length normalisation. Okapi BM25 originated in the Okapi information retrieval system.'
    },
    {
        doc_id: 'cmp_tfidf',
        title: 'TF-IDF Weighting',
        text:
            'TF-IDF weights a term by how often it appears in a document against how rare it is across the corpus. ' +
            'It shares the term frequency and inverse document frequency intuition with BM25 but applies no ' +
            'saturation and no length normalisation, so long documents score disproportionately highly.'
    },
    {
        doc_id: 'cmp_embeddings',
        title: 'Dense Embeddings',
        text:
            'Dense embeddings map text into a continuous vector space where distance reflects meaning. Two passages ' +
            'that share no vocabulary can sit close together if they express the same idea. Retrieval compares the ' +
            'query vector against document vectors using cosine similarity.'
    },
    {
        doc_id: 'cmp_cosine',
        title: 'Cosine Similarity',
        text:
            'Cosine similarity measures the angle between two vectors and ignores their magnitude, which makes it ' +
            'scale invariant. A long document and a short one on the same subject score alike. Euclidean distance ' +
            'behaves differently because magnitude contributes to the result.'
    },
    {
        doc_id: 'cmp_chunking',
        title: 'Chunking Strategy',
        text:
            'Splitting a document into overlapping passages keeps each embedding focused on a single idea. Without ' +
            'overlap a sentence spanning a boundary is cut in half and neither passage can answer a question about ' +
            'it. Chunk size trades precision against the amount of context each passage carries.'
    },
    {
        doc_id: 'cmp_rrf',
        title: 'Reciprocal Rank Fusion',
        text:
            'Reciprocal Rank Fusion merges ranked lists using only rank position, ignoring the underlying scores ' +
            'entirely. Each list contributes one over k plus rank. Because it never compares raw scores it is robust ' +
            'when the systems being merged produce values on incomparable scales.'
    },
    {
        doc_id: 'cmp_hybrid',
        title: 'Hybrid Retrieval',
        text:
            'Hybrid retrieval runs keyword and semantic search together and merges the outcomes. Keyword search ' +
            'handles rare exact terms and identifiers; semantic search handles paraphrase and intent. Weighted ' +
            'fusion normalises both score sets before blending them.'
    },
    {
        doc_id: 'cmp_rerank',
        title: 'Cross-Encoder Reranking',
        text:
            'A cross-encoder reads the query and a candidate passage together and scores their relevance jointly. ' +
            'This is far more accurate than comparing independent vectors but too slow to run across a whole corpus, ' +
            'so it is applied to a small candidate set retrieved by cheaper methods.'
    },
    {
        doc_id: 'cmp_pasta',
        title: 'Fresh Pasta',
        text:
            'Fresh pasta dough needs flour, eggs, and a pinch of salt. Knead until smooth, rest for thirty minutes, ' +
            'then roll thin and cut into ribbons. Boil in salted water for about three minutes until it floats.'
    },
    {
        doc_id: 'cmp_bread',
        title: 'Sourdough Bread',
        text:
            'Sourdough relies on a wild yeast starter rather than commercial yeast. Mix flour and water, let the ' +
            'dough autolyse, then fold it at intervals during a long bulk fermentation before shaping and baking.'
    }
];

interface Query {
    text: string;
    kind: 'exact-term' | 'conceptual' | 'mixed';
    note: string;
}

const QUERIES: Query[] = [
    { text: 'Okapi', kind: 'exact-term', note: 'rare proper noun, verbatim in one document' },
    { text: 'k1 saturation parameter', kind: 'exact-term', note: 'technical identifiers' },
    { text: 'autolyse', kind: 'exact-term', note: 'rare jargon, one document only' },
    { text: 'how do I prepare noodles from scratch', kind: 'conceptual', note: 'paraphrase, no shared vocabulary' },
    { text: 'why does vector length not affect the score', kind: 'conceptual', note: 'intent, not wording' },
    { text: 'merging two result lists without comparing scores', kind: 'conceptual', note: 'describes RRF without naming it' },
    { text: 'BM25 length normalisation', kind: 'mixed', note: 'exact term plus concept' },
    { text: 'splitting documents with overlap for retrieval', kind: 'mixed', note: 'partly verbatim' }
];

const POOL = 8;
const SHOW = 3;

// Read doc_id off the metadata rather than parsing the chunk id: ids are
// `{doc_id}_{hash}` and doc_ids themselves contain underscores, so splitting
// mangles them.
const docOf = (c: { metadata: Record<string, unknown> } | undefined) => String(c?.metadata?.doc_id ?? '-');
const label = (items: Array<{ metadata: Record<string, unknown> }>) =>
    items.length ? items.map(docOf).join(', ') : '(none)';

console.log('\nindexing comparison corpus...');
await guarded(() => ensureIndex());
for (const doc of CORPUS) {
    await guarded(() => indexDocument(doc.doc_id, doc.title, doc.text));
}
// Pinecone is eventually consistent; let the upserts land, then pull every
// chunk into BM25 so both arms see the same corpus.
await new Promise(resolve => setTimeout(resolve, 8000));
const hydrated = await guarded(() => rehydrateBm25());
console.log(`indexed ${CORPUS.length} documents; BM25 holds ${hydrated.documents} chunks.\n`);

console.log('=== strategy comparison ===');
console.log('single-arm rows call the retrievers directly, so they are genuinely isolated.\n');

let topOneDisagreements = 0;
let vectorOnlyFinds = 0;
let keywordOnlyFinds = 0;

for (const query of QUERIES) {
    const embedding = await guarded(() => embedQuery(query.text));

    // True isolation: the raw retrievers, not search() with a zeroed weight.
    const vectorRaw = await guarded(() => queryVectors(embedding, POOL));
    const bm25Raw = bm25.search(query.text, POOL);

    const vectorHits: ScoredCandidate[] = vectorRaw.map(h => ({ id: h.id, text: h.text, metadata: h.metadata, score: h.score }));
    const bm25Hits: ScoredCandidate[] = bm25Raw.map(h => ({ id: h.id, text: h.text, metadata: h.metadata, score: h.bm25_score }));

    const weighted = weightedFusion(vectorHits, bm25Hits, SHOW, 0.7, 0.3);
    const rrf = rrfFusion(vectorHits, bm25Hits, SHOW, 60);

    const vectorTop = vectorHits.slice(0, SHOW);
    const bm25Top = bm25Hits.slice(0, SHOW);

    const disagree = weighted[0]?.id !== rrf[0]?.id;
    if (disagree) {
        topOneDisagreements += 1;
    }

    // Which arm found the eventual winner? That is the concrete argument for
    // running both.
    const winner = weighted[0]?.id;
    const inVector = vectorHits.some(h => h.id === winner);
    const inBm25 = bm25Hits.some(h => h.id === winner);
    if (inVector && !inBm25) {
        vectorOnlyFinds += 1;
    }
    if (inBm25 && !inVector) {
        keywordOnlyFinds += 1;
    }

    console.log(`[${query.kind}] "${query.text}"`);
    console.log(`   ${query.note}`);
    console.log(`   vector-only : ${label(vectorTop)}`);
    console.log(`   bm25-only   : ${label(bm25Top)}${bm25Hits.length === 0 ? '   <- keyword search found nothing' : ''}`);
    console.log(`   weighted    : ${label(weighted)}`);
    console.log(`   rrf         : ${label(rrf)}`);
    console.log(`   top-1: weighted=${docOf(weighted[0])} rrf=${docOf(rrf[0])} ${disagree ? '<- DISAGREE' : ''}`);
    // Did fusion keep the answer vector search already had at #1?
    const vectorBest = docOf(vectorTop[0]);
    const keptByWeighted = weighted.slice(0, SHOW).some(r => docOf(r) === vectorBest);
    if (query.kind === 'conceptual' && !keptByWeighted) {
        console.log(`   NOTE: fusion dropped vector search's top hit (${vectorBest}) out of the top ${SHOW}`);
    }
    console.log(`   winner found by: ${[inVector && 'vector', inBm25 && 'bm25'].filter(Boolean).join(' + ') || 'neither'}`);
    console.log('');
}

console.log('--- summary ---');
console.log(`weighted vs RRF disagreed on top-1 for ${topOneDisagreements}/${QUERIES.length} queries.`);
console.log(`top result was reachable ONLY by vector search in ${vectorOnlyFinds} case(s).`);
console.log(`top result was reachable ONLY by keyword search in ${keywordOnlyFinds} case(s).`);
console.log('Those two counts are the case for hybrid: neither arm alone covers both.\n');

console.log('=== RRF k sweep ===');
console.log('k damps how sharply early ranks dominate. Watch the spread column shrink.\n');

for (const k of [20, 60, 100]) {
    console.log(`k = ${k}`);
    for (const query of QUERIES.slice(0, 4)) {
        const embedding = await guarded(() => embedQuery(query.text));
        const vectorRaw = await guarded(() => queryVectors(embedding, POOL));
        const vectorHits: ScoredCandidate[] = vectorRaw.map(h => ({ id: h.id, text: h.text, metadata: h.metadata, score: h.score }));
        const bm25Hits: ScoredCandidate[] = bm25.search(query.text, POOL).map(h => ({ id: h.id, text: h.text, metadata: h.metadata, score: h.bm25_score }));

        const fused = rrfFusion(vectorHits, bm25Hits, SHOW, k);
        const spread =
            fused.length > 1 ? (fused[0]!.hybrid_score - fused[fused.length - 1]!.hybrid_score).toFixed(5) : 'n/a';
        const bothLists = fused.filter(f => f.vector_rank !== null && f.bm25_rank !== null).length;

        console.log(
            `   "${query.text.slice(0, 40).padEnd(40)}" -> ${label(fused).padEnd(38)} ` +
                `top ${fused[0]?.hybrid_score.toFixed(5) ?? 'n/a'}, spread ${spread}, in-both ${bothLists}/${fused.length}`
        );
    }
    console.log('');
}
