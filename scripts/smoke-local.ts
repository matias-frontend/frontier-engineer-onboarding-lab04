/**
 * Everything testable without Pinecone or Gemini: chunking, BM25, and fusion.
 *
 * This is the suite to run while iterating. It needs no API keys, no network,
 * and no quota, and it covers the parts of the lab where the logic actually
 * lives — the retrieval maths rather than the plumbing.
 */
import { BM25Index } from '../src/bm25.js';
import { chunkDocument } from '../src/chunking.js';
import { minMaxNormalizer, rrfFusion, weightedFusion, type ScoredCandidate } from '../src/fusion.js';

let passed = 0;
let failed = 0;

function check(label: string, condition: boolean, detail?: unknown): void {
    if (condition) {
        passed += 1;
        console.log(`  PASS  ${label}`);
    } else {
        failed += 1;
        console.log(`  FAIL  ${label}${detail === undefined ? '' : `\n        ${JSON.stringify(detail)}`}`);
    }
}

const cand = (id: string, score: number): ScoredCandidate => ({ id, text: `text ${id}`, metadata: {}, score });

// ------------------------------------------------------------------ chunking
console.log('\nchunking');

const long = Array.from({ length: 40 }, (_, i) => `This is sentence number ${i} in a fairly long document.`).join(' ');
const chunks = chunkDocument(long, 300, 60);

check('produces multiple chunks', chunks.length > 1, chunks.length);
check('chunks respect the size ceiling', chunks.every(c => c.text.length <= 300), chunks.map(c => c.text.length));
check('chunk indexes are sequential', chunks.every((c, i) => c.chunk_index === i));
check('ids are unique', new Set(chunks.map(c => c.id)).size === chunks.length);
check('positions advance monotonically', chunks.every((c, i) => i === 0 || c.start_char > chunks[i - 1]!.start_char));

// Overlap is the point: consecutive chunks must actually share text.
const overlapping = chunks.length > 1 && chunks[1]!.start_char < chunks[0]!.end_char;
check('consecutive chunks overlap', overlapping, { first_end: chunks[0]?.end_char, second_start: chunks[1]?.start_char });

check('prefers sentence boundaries', chunks.slice(0, -1).filter(c => c.text.endsWith('.')).length > 0);
check('empty input yields no chunks', chunkDocument('   ').length === 0);
check('short input yields exactly one chunk', chunkDocument('Just a short sentence.', 500, 100).length === 1);
check('full text is covered', chunkDocument('abcdefghij', 4, 1).map(c => c.text).join('').includes('abcd'));

// A cursor that cannot advance would hang the process; it must throw instead.
let threw = false;
try {
    chunkDocument('some text here', 100, 100);
} catch {
    threw = true;
}
check('overlap >= chunkSize is rejected, not an infinite loop', threw);

// -------------------------------------------------------------------- BM25
console.log('\nBM25');

const index = new BM25Index();
index.addDocument('d1', 'Vector search uses embeddings to capture semantic meaning', { kind: 'vector' });
index.addDocument('d2', 'BM25 is a keyword ranking function based on term frequency', { kind: 'keyword' });
index.addDocument('d3', 'Hybrid search combines semantic embeddings with keyword ranking', { kind: 'hybrid' });
index.addDocument('d4', 'Cooking pasta requires boiling water and salt', { kind: 'unrelated' });

check('tracks document count', index.totalDocs === 4, index.totalDocs);
check('computes an average length', index.avgDocLength > 0, index.avgDocLength);

const bm25Pasta = index.search('pasta');
check('a term unique to one document ranks it first', bm25Pasta[0]?.id === 'd4', bm25Pasta.map(h => h.id));
check('scores are positive', (bm25Pasta[0]?.bm25_score ?? 0) > 0, bm25Pasta[0]?.bm25_score);
check('metadata is carried through', bm25Pasta[0]?.metadata.kind === 'unrelated', bm25Pasta[0]?.metadata);

check('a term in no document returns nothing', index.search('quantum').length === 0);
check('an empty query returns nothing', index.search('').length === 0);
check('a stop-words-only query returns nothing', index.search('the a of and').length === 0);

// IDF must favour the rare term: 'keyword' is in 2 docs, 'semantic' in 2, but
// 'BM25' is in exactly 1 and should dominate.
const rare = index.search('bm25');
check('rare terms rank their document first', rare[0]?.id === 'd2', rare.map(h => h.id));

const multi = index.search('semantic keyword');
check('documents matching more query terms rank higher', multi[0]?.id === 'd3', multi.map(h => h.id));

check('limit is respected', index.search('search', 1).length <= 1);

// Re-adding the same id must not corrupt df/avgdl.
const before = index.totalDocs;
index.addDocument('d1', 'Vector search uses embeddings to capture semantic meaning', { kind: 'vector' });
check('re-adding an id does not duplicate it', index.totalDocs === before, index.totalDocs);
check('re-added document is still findable', index.search('embeddings').some(h => h.id === 'd1'));

check('removing a document works', index.removeDocument('d4') && index.search('pasta').length === 0);
check('removing an unknown id returns false', !index.removeDocument('nope'));

const cleared = new BM25Index();
cleared.addDocument('x', 'hello world');
cleared.clear();
check('clear empties the index', cleared.totalDocs === 0 && cleared.search('hello').length === 0);

check('tokenizer strips stop words', !BM25Index.tokenize('the quick brown fox').includes('the'));
check('tokenizer lowercases', BM25Index.tokenize('HELLO World').includes('hello'));
check('tokenizer drops punctuation', BM25Index.tokenize('hello, world!').join(',') === 'hello,world');

// ---------------------------------------------------------------- fusion
console.log('\nnormalisation');

const norm = minMaxNormalizer([1, 3, 5]);
check('min maps to 0', norm(1) === 0);
check('max maps to 1', norm(5) === 1);
check('midpoint maps to 0.5', norm(3) === 0.5);
check('an all-equal set maps to 1, not 0', minMaxNormalizer([7, 7, 7])(7) === 1);
check('an empty set yields 0', minMaxNormalizer([])(1) === 0);

console.log('\nweighted fusion');

const vectorHits = [cand('a', 0.9), cand('b', 0.8), cand('c', 0.7)];
const bm25Hits = [cand('c', 12), cand('d', 6)];

const weighted = weightedFusion(vectorHits, bm25Hits, 10, 0.7, 0.3);
check('unions both lists', weighted.length === 4, weighted.map(r => r.id));
check('sorted by hybrid_score descending', weighted.every((r, i) => i === 0 || weighted[i - 1]!.hybrid_score >= r.hybrid_score));
check('carries both component scores', weighted.every(r => typeof r.vector_score === 'number' && typeof r.bm25_score === 'number'));

const cEntry = weighted.find(r => r.id === 'c')!;
const aEntry = weighted.find(r => r.id === 'a')!;
check('an item in both lists gets both contributions', cEntry.vector_score > 0 && cEntry.bm25_score > 0, cEntry);
check('an item in both lists records both ranks', cEntry.vector_rank !== null && cEntry.bm25_rank !== null, cEntry);

// Min-max maps the WORST item in a list to exactly 0. `c` is last among the
// vector hits, so its vector contribution vanishes entirely and only its BM25
// share (0.3) survives — while `a`, top of the vector list, keeps the full 0.7.
// Appearing in both lists therefore does NOT guarantee a higher weighted score.
// This is the concrete weakness RRF avoids; see the comparison below.
check('worst-in-list normalises to zero', cEntry.hybrid_score === 0.3, cEntry.hybrid_score);
check('top vector hit keeps its full weight', aEntry.hybrid_score === 0.7, aEntry.hybrid_score);
check(
    'weighted fusion can rank a single-list hit above a both-list hit',
    aEntry.hybrid_score > cEntry.hybrid_score,
    { a: aEntry.hybrid_score, c: cEntry.hybrid_score }
);

const vectorOnly = weightedFusion(vectorHits, bm25Hits, 10, 1, 0);
check('vector_weight=1 ranks by vector score alone', vectorOnly[0]!.id === 'a', vectorOnly.map(r => r.id));
const keywordOnly = weightedFusion(vectorHits, bm25Hits, 10, 0, 1);
check('bm25_weight=1 ranks by keyword score alone', keywordOnly[0]!.id === 'c', keywordOnly.map(r => r.id));

check('limit is respected', weightedFusion(vectorHits, bm25Hits, 2).length === 2);
check('empty inputs yield no results', weightedFusion([], [], 10).length === 0);
check('one empty side still returns the other', weightedFusion(vectorHits, [], 10).length === 3);

console.log('\nRRF fusion');

const rrf = rrfFusion(vectorHits, bm25Hits, 10, 60);
check('unions both lists', rrf.length === 4, rrf.map(r => r.id));
check('sorted by score descending', rrf.every((r, i) => i === 0 || rrf[i - 1]!.hybrid_score >= r.hybrid_score));
check('an item in both lists ranks first', rrf[0]!.id === 'c', rrf.map(r => r.id));

// RRF depends only on rank, so wildly different score scales must not matter.
const rescaled = rrfFusion(
    [cand('a', 0.0009), cand('b', 0.0008), cand('c', 0.0007)],
    [cand('c', 99999), cand('d', 50000)],
    10,
    60
);
check('identical ranks produce identical output regardless of scale',
    JSON.stringify(rescaled.map(r => r.id)) === JSON.stringify(rrf.map(r => r.id)),
    { rescaled: rescaled.map(r => r.id), original: rrf.map(r => r.id) });

// `c` sits at vector rank 2 and BM25 rank 0, so its score is 1/63 + 1/61.
const expectedC = 1 / (60 + 2 + 1) + 1 / (60 + 0 + 1);
check('score matches the RRF formula exactly', Math.abs(rrf[0]!.hybrid_score - expectedC) < 1e-12, {
    got: rrf[0]!.hybrid_score,
    expected: expectedC
});

// The payoff: on the same inputs, RRF promotes the both-list item that
// weighted fusion demoted to third. RRF sees rank, not raw score, so being
// last in one list still contributes 1/63 instead of being zeroed out.
const weightedOrder = weighted.map(r => r.id);
const rrfOrder = rrf.map(r => r.id);
check(
    'RRF ranks the both-list item first where weighted did not',
    rrfOrder[0] === 'c' && weightedOrder[0] !== 'c',
    { weighted: weightedOrder, rrf: rrfOrder }
);

// Smaller k sharpens the advantage of early ranks.
const sharp = rrfFusion(vectorHits, bm25Hits, 10, 20);
const flat = rrfFusion(vectorHits, bm25Hits, 10, 100);
const spread = (list: typeof rrf) => list[0]!.hybrid_score - list[list.length - 1]!.hybrid_score;
check('smaller k favours early ranks more sharply', spread(sharp) > spread(flat), { k20: spread(sharp), k100: spread(flat) });

check('empty inputs yield no results', rrfFusion([], [], 10).length === 0);

console.log(`\n${passed} passed, ${failed} failed\n`);
process.exit(failed === 0 ? 0 : 1);
