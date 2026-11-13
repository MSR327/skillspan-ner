// Core extraction logic, shared by index.html and the Node parity test (scripts/js_parity.mjs).
// Same word/sentence splitting as skillex/text.py (matches SkillSpan's training tokens).
const WORD = /[\p{L}\p{N}_\p{M}]+(?:['’.\-][\p{L}\p{N}_\p{M}]+)*|\S/gu;
export const splitWords = (text) => [...text.matchAll(WORD)].map((m) => [m.index, m.index + m[0].length]);
export function splitSentences(text, words) {
  const sents = []; let cur = [];
  words.forEach(([a, b], i) => {
    if (cur.length && text.slice(words[cur.at(-1)][1], a).includes("\n")) { sents.push(cur); cur = []; }
    cur.push(i);
    if ([".", "!", "?"].includes(text.slice(a, b))) { sents.push(cur); cur = []; }
  });
  if (cur.length) sents.push(cur);
  return sents;
}

// Viterbi with the learned CRF transitions (identical to skillex.decode.viterbi_bio).
export function viterbi(emit, trans, start) {
  const T = emit.length; if (!T) return [];
  let best = [0, 1, 2].map((j) => start[j] + emit[0][j]);
  const back = [];
  for (let t = 1; t < T; t++) {
    const bt = [], nb = [];
    for (let j = 0; j < 3; j++) {
      let arg = 0, val = -Infinity;
      for (let i = 0; i < 3; i++) { const v = best[i] + trans[i][j]; if (v > val) { val = v; arg = i; } }
      bt.push(arg); nb.push(val + emit[t][j]);
    }
    back.push(bt); best = nb;
  }
  let k = best.indexOf(Math.max(...best)); const path = [k];
  for (let t = T - 1; t > 0; t--) { k = back[t - 1][k]; path.push(k); }
  return path.reverse();
}


const MAX_SUBWORDS = 510, BATCH = 16, TYPES = ["SKILL", "KNOWLEDGE"];

// Split a sentence (list of global word ids) into chunks that fit the model.
function encodeSentence(tok, CLS, SEP, text, words, sent) {
  const chunks = []; let ids = [CLS], first = [], wids = [];
  for (const w of sent) {
    // Leading space = SentencePiece word-start marker "▁". Python's is_split_into_words adds it;
    // transformers.js does not (without it, IDs differ for every word: 0/897 matched).
    const sub = tok.encode(" " + text.slice(...words[w]), { add_special_tokens: false });
    if (ids.length + sub.length > MAX_SUBWORDS && wids.length) {
      chunks.push({ ids: [...ids, SEP], first, wids }); ids = [CLS]; first = []; wids = [];
    }
    if (!sub.length) continue;
    first.push(ids.length); wids.push(w); ids.push(...sub.slice(0, MAX_SUBWORDS - 1));
  }
  if (wids.length) chunks.push({ ids: [...ids, SEP], first, wids });
  return chunks;
}

export async function extract({ ort, tok, session, crfs }, text) {
  const [CLS, SEP] = tok.encode("", { add_special_tokens: true });
  const words = splitWords(text);
  const chunks = splitSentences(text, words).flatMap((s) => encodeSentence(tok, CLS, SEP, text, words, s));
  const spans = [];
  for (let b = 0; b < chunks.length; b += BATCH) {
    const batch = chunks.slice(b, b + BATCH), L = Math.max(...batch.map((c) => c.ids.length));
    const ids = new BigInt64Array(batch.length * L), am = new BigInt64Array(batch.length * L);
    batch.forEach((c, r) => c.ids.forEach((v, j) => { ids[r * L + j] = BigInt(v); am[r * L + j] = 1n; }));
    const out = await session.run({
      input_ids: new ort.Tensor("int64", ids, [batch.length, L]),
      attention_mask: new ort.Tensor("int64", am, [batch.length, L]),
    });
    const lg = out.logits.data;
    batch.forEach((c, r) => {
      TYPES.forEach((type, h) => {
        const emit = c.first.map((p) => [0, 1, 2].map((k) => lg[(r * L + p) * 6 + h * 3 + k]));
        const path = viterbi(emit, crfs[h].transitions, crfs[h].start);
        let s = -1;
        for (let i = 0; i <= path.length; i++) {
          const tag = i < path.length ? path[i] : 0;  // 0=O 1=B 2=I
          if (s >= 0 && tag !== 2) { spans.push({ type, start: words[c.wids[s]][0], end: words[c.wids[i - 1]][1] }); s = -1; }
          if (tag === 1 || (tag === 2 && s < 0)) s = i;
        }
      });
    });
  }
  return spans;
}

