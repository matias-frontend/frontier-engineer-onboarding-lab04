/**
 * Extension challenge 1: RRF vs weighted scoring, and hybrid vs each arm alone.
 *
 * Runs a fixed query set through five configurations and prints where they
 * disagree. The point is not to crown a winner — it is to show *which query
 * types* each strategy handles best, which is what the lab asks for.
 *
 * Also sweeps RRF's k (20 / 60 / 100) as the extra exercise suggests.
 *
 * Needs PINECONE_API_KEY and GOOGLE_API_KEY, and assumes the corpus from
 * smoke-e2e.ts is already indexed.
 */
import { ensureIndex } from '../src/pinecone.js';
import { isDailyQuotaExhausted, quotaGuidance } from '../src/quota.js';
import { search } from '../src/search.js';

for (const key of ['PINECONE_API_KEY', 'GOOGLE_API_KEY']) {
    if (!process.env[key]) {
        console.error(`\n  ${key} is not set.\n`);
        process.exit(1);
    }
}

interface Query {
    text: string;
    kind: 'exact-term' | 'conceptual' | 'mixed';
    note: string;
}

const QUERIES: Query[] = [
    { text: 'Okapi', kind: 'exact-term', note: 'rare proper noun, appears verbatim in one doc' },
    { text: 'k1 parameter saturation', kind: 'exact-term', note: 'technical identifiers' },
    { text: 'how do I prepare noodles from scratch', kind: 'conceptual', note: 'paraphrase, no shared keywords' },
    { text: 'why combine two retrieval methods', kind: 'conceptual', note: 'intent, not wording' },
    { text: 'BM25 ranking for keyword retrieval', kind: 'mixed', note: 'exact term plus concept' },
    { text: 'document length normalisation in scoring', kind: 'mixed', note: 'phrase partly verbatim' }
];

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

await guarded(() => ensureIndex());

const label = (ids: string[]) => (ids.length ? ids.join(', ') : '(none)');
const docs = (results: Array<{ metadata: Record<string, unknown> }>) =>
    results.map(r => String(r.metadata.doc_id ?? '?'));

console.log('\n=== strategy comparison ===\n');

let agreements = 0;

for (const query of QUERIES) {
    const [vectorOnly, keywordOnly, weighted, rrf] = await Promise.all([
        guarded(() => search(query.text, { limit: 3, vectorWeight: 1, bm25Weight: 0 })),
        guarded(() => search(query.text, { limit: 3, vectorWeight: 0, bm25Weight: 1 })),
        guarded(() => search(query.text, { limit: 3, method: 'weighted' })),
        guarded(() => search(query.text, { limit: 3, method: 'rrf' }))
    ]);

    const w = docs(weighted.results);
    const r = docs(rrf.results);
    const sameTop = w[0] === r[0];
    if (sameTop) {
        agreements += 1;
    }

    console.log(`[${query.kind}] "${query.text}"`);
    console.log(`   ${query.note}`);
    console.log(`   vector-only : ${label(docs(vectorOnly.results))}`);
    console.log(`   bm25-only   : ${label(docs(keywordOnly.results))}`);
    console.log(`   weighted    : ${label(w)}`);
    console.log(`   rrf         : ${label(r)}`);
    console.log(`   top-1 agreement between weighted and rrf: ${sameTop ? 'yes' : 'NO'}`);
    console.log('');
}

console.log(`weighted and RRF agreed on top-1 for ${agreements}/${QUERIES.length} queries.\n`);

console.log('=== RRF k sweep ===\n');
console.log('k damps how sharply early ranks dominate: small k rewards being #1 in one list,');
console.log('large k rewards appearing in both lists at all.\n');

for (const k of [20, 60, 100]) {
    console.log(`k = ${k}`);
    for (const query of QUERIES.slice(0, 4)) {
        const result = await guarded(() => search(query.text, { limit: 3, method: 'rrf', k }));
        const top = result.results[0];
        const spread =
            result.results.length > 1
                ? (result.results[0]!.hybrid_score - result.results[result.results.length - 1]!.hybrid_score).toFixed(5)
                : 'n/a';
        console.log(`   "${query.text.slice(0, 42)}" -> ${label(docs(result.results))}  (top score ${top?.hybrid_score.toFixed(5) ?? 'n/a'}, spread ${spread})`);
    }
    console.log('');
}

console.log('Read the spread column: it shrinks as k grows, which is the damping working.\n');
