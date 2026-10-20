// LectureIQ core for the browser: a JavaScript port of src/lectureiq's transcript parsing,
// chunking, BM25 retrieval, offline engine, and exports. Keep it in step with the Python
// package; tests/test_demo_port.py grades it with the same evalkit QA suite.
(function (root) {
  "use strict";

  // ---------- text ----------

  const STOPWORDS = new Set(`
    a about above after again against all also am an and any are aren't as at be because been
    before being below between both but by can can't cannot could did do does doing don't down
    during each few for from further get gets got had has have having he her here hers herself
    him himself his how i if in into is isn't it it's its itself just let's like me more most my
    myself no nor not now of off on once one only or other ought our ours ourselves out over own
    really right same she should so some such than that that's the their theirs them themselves
    then there there's these they this those through to too under until up us very was we were
    what when where which while who whom why will with would you your yours yourself yourselves
    okay ok um uh yeah well going gonna kind sort thing things lot lots way today see actually
    basically maybe mean know think say said make makes made want use used using able need
  `.split(/\s+/).filter(Boolean));

  const WORD = /[A-Za-z][A-Za-z\-']+/g;
  const SENTENCE_END = /(?<=[.!?])\s+(?=[A-Z0-9"'(])/;

  const words = (text) => (text.match(WORD) || []).map((w) => w.toLowerCase().replace(/^['-]+|['-]+$/g, ""));
  const contentWords = (text) => words(text).filter((w) => !STOPWORDS.has(w) && w.length > 2);
  const sentences = (text) => text.split(SENTENCE_END).map((s) => s.trim()).filter((s) => s.split(/\s+/).filter(Boolean).length >= 4);
  const wordCount = (text) => text.split(/\s+/).filter(Boolean).length;
  const squash = (text) => text.split(/\s+/).filter(Boolean).join(" ");

  function counter(items) {
    const m = new Map();
    for (const it of items) m.set(it, (m.get(it) || 0) + 1);
    return m;
  }
  function mostCommon(m, n) {
    return [...m.entries()].sort((a, b) => b[1] - a[1]).slice(0, n).map(([w]) => w);
  }
  const keywords = (text, n = 10) => mostCommon(counter(contentWords(text)), n);

  // ---------- transcripts ----------

  const MAX_CHARS = 2000000;
  const TS = String.raw`(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})`;
  const CUE_TIMING = new RegExp(String.raw`^\s*${TS}\s*-->\s*${TS}`);
  const TAG = /<[^>]+>/g;

  class TranscriptError extends Error {}

  function formatTs(seconds) {
    const total = Math.floor(seconds);
    const h = Math.floor(total / 3600), m = Math.floor((total % 3600) / 60), s = total % 60;
    const ss = String(s).padStart(2, "0");
    return h ? `${h}:${String(m).padStart(2, "0")}:${ss}` : `${m}:${ss}`;
  }

  const toSeconds = (g) => Number(g[0] || 0) * 3600 + Number(g[1] || 0) * 60 + Number(g[2] || 0) + Number(g[3] || 0) / 1000;

  function parseCues(text) {
    const segments = [];
    const lines = text.split(/\r?\n/);
    let i = 0;
    while (i < lines.length) {
      const m = CUE_TIMING.exec(lines[i]);
      if (!m) { i += 1; continue; }
      const start = toSeconds(m.slice(1, 5)), end = toSeconds(m.slice(5, 9));
      i += 1;
      const body = [];
      while (i < lines.length && lines[i].trim()) { body.push(lines[i].replace(TAG, "").trim()); i += 1; }
      const cue = body.filter(Boolean).join(" ");
      if (cue) segments.push({ text: cue, start, end });
    }
    return segments;
  }

  function parsePlain(text) {
    return text.split(/\n\s*\n/).map(squash).filter(Boolean).map((p) => ({ text: p, start: null, end: null }));
  }

  function parseTranscript(text, title = "Untitled lecture") {
    if (text.length > MAX_CHARS) throw new TranscriptError(`transcript is too large (max ${MAX_CHARS.toLocaleString()} characters)`);
    text = text.replace(/^﻿/, "");
    const cued = text.split(/\r?\n/).some((line) => CUE_TIMING.test(line));
    const segments = cued ? parseCues(text) : parsePlain(text);
    if (!segments.length) throw new TranscriptError("transcript has no text");
    return { title: title.trim() || "Untitled lecture", segments, chunks: [] };
  }

  const lectureText = (lec) => lec.segments.map((s) => s.text).join(" ");
  const hasTimestamps = (lec) => lec.segments.some((s) => s.start != null);

  function splitLong(seg, maxWords) {
    if (wordCount(seg.text) <= maxWords) return [seg];
    const pieces = [];
    let current = [];
    for (const sentence of seg.text.split(/(?<=[.!?])\s+/)) {
      if (current.length && wordCount([...current, sentence].join(" ")) > maxWords) {
        pieces.push({ text: current.join(" "), start: seg.start, end: seg.end });
        current = [];
      }
      current.push(sentence);
    }
    if (current.length) pieces.push({ text: current.join(" "), start: seg.start, end: seg.end });
    return pieces;
  }

  function chunkLecture(lec, maxWords = 300) {
    if (maxWords < 20) throw new Error("maxWords must be at least 20");
    const chunks = [];
    let group = [], n = 0;
    const flush = () => {
      if (group.length) {
        const starts = group.map((s) => s.start).filter((v) => v != null);
        const ends = group.map((s) => s.end).filter((v) => v != null);
        chunks.push({
          id: chunks.length + 1,
          text: group.map((s) => s.text).join(" "),
          start: starts.length ? Math.min(...starts) : null,
          end: ends.length ? Math.max(...ends) : null,
          segments: group,
        });
      }
      group = []; n = 0;
    };
    for (const original of lec.segments) {
      for (const seg of splitLong(original, maxWords)) {
        const w = wordCount(seg.text);
        if (group.length && n + w > maxWords) flush();
        group.push(seg);
        n += w;
      }
    }
    flush();
    lec.chunks = chunks;
    return chunks;
  }

  function timeOf(chunk, quote) {
    const probe = squash(quote).slice(0, 60);
    for (const seg of chunk.segments || []) {
      if (seg.start != null && (seg.text.includes(probe) || quote.includes(seg.text))) return seg.start;
    }
    return chunk.start;
  }

  // ---------- retrieval ----------

  function bm25Search(chunks, query, k = 4, k1 = 1.5, b = 0.75) {
    const tfs = chunks.map((c) => counter(contentWords(c.text)));
    const lens = tfs.map((tf) => [...tf.values()].reduce((a, v) => a + v, 0));
    const avg = lens.length ? lens.reduce((a, v) => a + v, 0) / lens.length : 0;
    const df = new Map();
    for (const tf of tfs) for (const t of tf.keys()) df.set(t, (df.get(t) || 0) + 1);
    const n = chunks.length;
    const idf = (t) => Math.log(1 + (n - df.get(t) + 0.5) / (df.get(t) + 0.5));
    const terms = contentWords(query);
    const scored = chunks.map((c, i) => {
      let total = 0;
      for (const t of terms) {
        const f = tfs[i].get(t) || 0;
        if (f) total += idf(t) * f * (k1 + 1) / (f + k1 * (1 - b + b * lens[i] / (avg || 1)));
      }
      return [c, total];
    });
    return scored.sort((x, y) => y[1] - x[1]).slice(0, k).filter(([, s]) => s > 0);
  }

  // ---------- offline engine ----------

  const DEFINITION = /^(?:(?:so|now|and|well|okay|remember)[, ]+)*(?:an?\s+|the\s+)?(?<term>[A-Za-z][\w\-]*(?:\s+[\w\-]+){0,3}?)\s*,?\s+(?:is|are|refers to|is defined as|means|describes)\s+(?<definition>.{12,})$/i;
  const NOT_TERMS = new Set([...STOPWORDS, "everything", "something", "nothing", "anything", "answer", "idea"]);
  const MAX_POINT_WORDS = 45, MIN_DEFINITION_WORDS = 4, DEFINITION_BONUS = 1.3, TLDR_SENTENCES = 4, DEFINES_TERM_BONUS = 2;

  function crc32(str) {
    let c, crc = 0xffffffff;
    for (let i = 0; i < str.length; i++) {
      c = (crc ^ str.charCodeAt(i)) & 0xff;
      for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
      crc = (crc >>> 8) ^ c;
    }
    return (crc ^ 0xffffffff) >>> 0;
  }
  function rng(seed) { // mulberry32: deterministic per item, so a lecture always gets the same quiz
    let a = crc32(seed);
    return () => {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function shuffle(arr, rand) {
    for (let i = arr.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [arr[i], arr[j]] = [arr[j], arr[i]]; }
    return arr;
  }
  const sample = (arr, k, rand) => shuffle(arr.slice(), rand).slice(0, k);

  function score(sentence, freq) {
    const tokens = contentWords(sentence);
    if (!tokens.length) return 0;
    let s = 0;
    for (const t of new Set(tokens)) s += freq.get(t) || 0;
    s /= Math.pow(tokens.length, 0.6);
    return DEFINITION.test(sentence) ? s * DEFINITION_BONUS : s;
  }

  function findTerms(chunks) {
    const found = new Map();
    for (const chunk of chunks) {
      for (const sentence of sentences(chunk.text)) {
        const m = DEFINITION.exec(sentence.trim());
        if (!m) continue;
        const term = m.groups.term.trim();
        const parts = term.split(/\s+/);
        if (NOT_TERMS.has(parts[0].toLowerCase()) || parts.every((w) => NOT_TERMS.has(w.toLowerCase()))) continue;
        const definition = m.groups.definition.trim().replace(/\.+$/, "");
        if (wordCount(definition) < MIN_DEFINITION_WORDS) continue;
        const key = term.toLowerCase();
        if (!found.has(key)) {
          found.set(key, {
            term: term[0].toUpperCase() + term.slice(1),
            definition: definition[0].toUpperCase() + definition.slice(1),
            chunk_id: chunk.id,
            start: timeOf(chunk, sentence),
          });
        }
      }
    }
    return [...found.values()];
  }

  function blank(sentence, freq) {
    const candidates = [...new Set(contentWords(sentence))].sort((a, b) => (freq.get(b) || 0) - (freq.get(a) || 0));
    for (const word of candidates) {
      const pattern = new RegExp(`\\b${word.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b`, "i");
      const m = pattern.exec(sentence);
      if (m) return [sentence.replace(pattern, "_____"), m[0]];
    }
    return null;
  }

  const DEFAULT_OPTIONS = { maxKeyPoints: 10, maxTerms: 25, maxFlashcards: 20, maxQuiz: 10 };

  const offlineEngine = {
    name: "offline",

    studyPack(lec, chunks, options = DEFAULT_OPTIONS) {
      const freq = counter(contentWords(lectureText(lec)));
      const byId = new Map(chunks.map((c) => [c.id, c]));
      const scored = [];
      for (const chunk of chunks) for (const s of sentences(chunk.text)) {
        if (wordCount(s) <= MAX_POINT_WORDS) scored.push({ score: score(s, freq), cid: chunk.id, s, order: scored.length });
      }
      const best = scored.slice().sort((a, b) => b.score - a.score);
      const inOrder = (items) => items.slice().sort((a, b) => a.order - b.order);
      const keyPoints = inOrder(best.slice(0, options.maxKeyPoints)).map((t) => ({ text: t.s, chunk_id: t.cid, start: timeOf(byId.get(t.cid), t.s) }));
      const perChunk = new Map();
      for (const item of best) if (!perChunk.has(item.cid)) perChunk.set(item.cid, item);
      const leads = [...perChunk.values()].sort((a, b) => b.score - a.score).slice(0, TLDR_SENTENCES);
      const tldr = inOrder(leads).map((t) => t.s).join(" ") || lectureText(lec).slice(0, 400);
      const terms = findTerms(chunks).slice(0, options.maxTerms);
      const sections = chunks.map((chunk) => {
        const top = keywords(chunk.text, 3);
        const ranked = sentences(chunk.text).sort((a, b) => score(a, freq) - score(b, freq));
        return {
          heading: top.map((w) => w[0].toUpperCase() + w.slice(1)).join(" · ") || `Part ${chunk.id}`,
          summary: ranked.length ? ranked.slice(-2).join(" ") : chunk.text.slice(0, 300),
          chunk_ids: [chunk.id],
          start: chunk.start,
        };
      });
      return {
        title: lec.title,
        tldr,
        sections,
        key_points: keyPoints,
        terms,
        flashcards: this._flashcards(terms, keyPoints, freq).slice(0, options.maxFlashcards),
        quiz: this._quiz(terms, keyPoints, freq).slice(0, options.maxQuiz),
        engine: "offline",
        warnings: terms.length ? [] : ["No definitional sentences found; glossary is empty."],
      };
    },

    _flashcards(terms, points, freq) {
      const cards = terms.map((t) => ({ front: `What is meant by “${t.term}”?`, back: t.definition, chunk_id: t.chunk_id, start: t.start }));
      for (const p of points) {
        const b = blank(p.text, freq);
        if (b) cards.push({ front: `Fill in the blank: ${b[0]}`, back: b[1], chunk_id: p.chunk_id, start: p.start });
      }
      return cards;
    },

    _quiz(terms, points, freq) {
      const quiz = [];
      if (terms.length >= 2) {
        for (const term of terms) {
          const rand = rng(term.term);
          const others = terms.filter((t) => t !== term).map((t) => t.definition);
          const options = shuffle([term.definition, ...sample(others, Math.min(3, others.length), rand)], rand);
          quiz.push({
            question: `Which statement best describes “${term.term}”?`,
            options,
            answer: options.indexOf(term.definition),
            explanation: `${term.term}: ${term.definition}.`,
            chunk_id: term.chunk_id,
            start: term.start,
          });
        }
      }
      const vocabulary = mostCommon(freq, 40);
      for (const p of points) {
        const b = blank(p.text, freq);
        if (!b) continue;
        const [sentence, word] = b;
        const rand = rng(sentence);
        const pool = vocabulary.filter((w) => w !== word.toLowerCase());
        const options = [word, ...sample(pool, Math.min(3, pool.length), rand)];
        if (options.length < 2) continue;
        shuffle(options, rand);
        quiz.push({
          question: `Complete the statement: ${sentence}`,
          options,
          answer: options.indexOf(word),
          explanation: p.text,
          chunk_id: p.chunk_id,
          start: p.start,
        });
      }
      return quiz;
    },

    answer(question, passages) {
      const wanted = new Set(contentWords(question));
      const relevance = (s) => {
        const overlap = contentWords(s).filter((w, i, all) => all.indexOf(w) === i && wanted.has(w)).length;
        const m = DEFINITION.exec(s.trim());
        if (overlap && m) {
          const term = new Set(contentWords(m.groups.term));
          if (term.size && [...term].every((w) => wanted.has(w))) return overlap + DEFINES_TERM_BONUS;
        }
        return overlap;
      };
      const ranked = [];
      passages.forEach((chunk, rank) => {
        const ss = sentences(chunk.text);
        for (const s of ss.length ? ss : [chunk.text]) ranked.push({ rel: relevance(s), rank, chunk, s });
      });
      ranked.sort((a, b) => b.rel - a.rel || a.rank - b.rank);
      const hits = ranked.slice(0, 2).filter((h) => h.rel);
      if (!hits.length) return { question, answer: "The lecture doesn't seem to cover that.", citations: [], grounded: false };
      const citations = new Map();
      for (const h of hits) if (!citations.has(h.chunk.id)) citations.set(h.chunk.id, { chunk_id: h.chunk.id, start: timeOf(h.chunk, h.s), quote: h.s });
      return { question, answer: hits.map((h) => h.s).join(" "), citations: [...citations.values()], grounded: true };
    },
  };

  const NOT_COVERED = "The lecture doesn't seem to cover that.";

  async function ask(lec, question, engine, k = 4) {
    question = question.trim();
    if (!question) throw new Error("question is empty");
    if (!lec.chunks.length) chunkLecture(lec);
    const passages = bm25Search(lec.chunks, question, k).map(([c]) => c);
    if (!passages.length) return { question, answer: NOT_COVERED, citations: [], grounded: false };
    return engine.answer(question, passages);
  }

  // ---------- exports ----------

  const at = (start) => (start != null ? ` (${formatTs(start)})` : "");

  function toMarkdown(pack) {
    const out = [`# ${pack.title}`, "", `> ${pack.tldr}`, ""];
    if (pack.sections.length) {
      out.push("## Outline", "");
      pack.sections.forEach((s, i) => out.push(`${i + 1}. **${s.heading}**${at(s.start)}: ${s.summary}`));
      out.push("");
    }
    if (pack.key_points.length) out.push("## Key points", "", ...pack.key_points.map((p) => `- ${p.text}`), "");
    if (pack.terms.length) out.push("## Glossary", "", ...pack.terms.map((t) => `- **${t.term}**: ${t.definition}`), "");
    if (pack.flashcards.length) out.push("## Flashcards", "", ...pack.flashcards.map((c) => `- **Q:** ${c.front}  \n  **A:** ${c.back}`), "");
    if (pack.quiz.length) {
      out.push("## Quiz", "");
      pack.quiz.forEach((q, i) => {
        out.push(`${i + 1}. ${q.question}`);
        q.options.forEach((o, j) => out.push(`   - ${String.fromCharCode(65 + j)}. ${o}`));
      });
      out.push("", "### Answers", "");
      pack.quiz.forEach((q, i) => out.push(`${i + 1}. **${String.fromCharCode(65 + q.answer)}**${q.explanation ? `: ${q.explanation}` : ""}`));
      out.push("");
    }
    out.push(`_Generated by LectureIQ (${pack.engine} engine)._`);
    return out.join("\n") + "\n";
  }

  function flashcardsCsv(pack) {
    const cell = (v) => (/[",\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v);
    return pack.flashcards.map((c) => `${cell(c.front)},${cell(c.back)}`).join("\n") + (pack.flashcards.length ? "\n" : "");
  }

  const api = {
    STOPWORDS, words, contentWords, sentences, keywords,
    TranscriptError, parseTranscript, chunkLecture, formatTs, timeOf, lectureText, hasTimestamps, wordCount,
    bm25Search, offlineEngine, ask, NOT_COVERED, toMarkdown, flashcardsCsv, DEFAULT_OPTIONS,
  };
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.LectureIQ = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
