// LectureIQ frontend. Vanilla JS, no build step. All lecture-derived text is inserted with
// textContent (via h()), never innerHTML, so transcripts cannot inject markup.
"use strict";

const $ = (sel) => document.querySelector(sel);

function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") el.className = value;
    else if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
    else if (value !== false && value != null) el.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child == null || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

function fmtTime(seconds) {
  const s = Math.floor(seconds);
  const hh = Math.floor(s / 3600);
  const mm = Math.floor((s % 3600) / 60);
  const ss = String(s % 60).padStart(2, "0");
  return hh ? `${hh}:${String(mm).padStart(2, "0")}:${ss}` : `${mm}:${ss}`;
}

async function api(path, options = {}) {
  const res = await fetch(path, options);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error || `Request failed (${res.status})`);
  return body;
}

const state = { lecture: null, cards: [], cardIndex: 0, flipped: false, audio: false, dropLabel: "" };
const AUDIO_EXT = /\.(flac|m4a|mp3|mp4|mpeg|mpga|oga|ogg|wav|webm)$/i;

// ---------- intake ----------

const status = (msg) => { $("#status").textContent = msg; };

function setBusy(busy) {
  for (const id of ["#analyze", "#sample"]) $(id).disabled = busy;
}

async function submit(promise, message = "Analyzing… this can take a minute with an LLM engine.") {
  setBusy(true);
  status(message);
  try {
    show(await promise);
    status("");
    loadRecent();
  } catch (err) {
    status(err.message);
  } finally {
    setBusy(false);
  }
}

$("#intake-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const text = $("#text").value.trim();
  if (!text) { status("Paste a transcript or choose a file first."); return; }
  submit(api("/api/lectures", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, title: $("#title").value.trim() }),
  }));
});

$("#sample").addEventListener("click", () => submit(api("/api/lectures/sample", { method: "POST" })));

function uploadFile(file) {
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  form.append("title", $("#title").value.trim());
  $("#drop-label").textContent = file.name;
  const audio = AUDIO_EXT.test(file.name);
  if (audio && !state.audio) {
    status("This server doesn't transcribe audio; start it with --transcribe.");
    return;
  }
  submit(api("/api/lectures", { method: "POST", body: form }),
    audio ? "Transcribing and analyzing… long recordings take a few minutes." : undefined);
}

const drop = $("#drop");
drop.addEventListener("click", () => $("#file").click());
drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") $("#file").click(); });
$("#file").addEventListener("change", (e) => uploadFile(e.target.files[0]));
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("over");
  uploadFile(e.dataTransfer.files[0]);
});

const DROP_LABEL = $("#drop-label").textContent;

$("#new").addEventListener("click", () => {
  $("#results").hidden = true;
  $("#intake").hidden = false;
  $("#file").value = "";
  $("#drop-label").textContent = state.dropLabel || DROP_LABEL;
  loadRecent();
  history.replaceState(null, "", location.pathname);
  $("#text").focus();
});

// ---------- results ----------

function tsButton(chunkId, start) {
  if (chunkId == null) return null;
  const label = start != null ? `▶ ${fmtTime(start)}` : `§${chunkId}`;
  return h("button", {
    class: "ts", type: "button", title: "Show in transcript",
    onclick: () => jumpTo(chunkId),
  }, label);
}

function chunkStart(chunkId) {
  const chunk = state.lecture.chunks.find((c) => c.id === chunkId);
  return chunk ? chunk.start : null;
}

// Items carry their own time when the engine knows the exact sentence; else the chunk's.
const itemStart = (item) => (item.start != null ? item.start : chunkStart(item.chunk_id));

function show(lecture) {
  state.lecture = lecture;
  const pack = lecture.pack;
  $("#intake").hidden = true;
  $("#results").hidden = false;
  history.replaceState(null, "", `#${lecture.id}`);

  $("#pack-title").textContent = pack.title;
  $("#pack-meta").textContent = [
    `${lecture.word_count.toLocaleString()} words`,
    `${lecture.chunks.length} part${lecture.chunks.length === 1 ? "" : "s"}`,
    lecture.has_timestamps ? "timestamped" : "no timestamps",
    `${pack.engine} engine`,
  ].join(" · ");
  $("#tldr").textContent = pack.tldr;
  $("#dl-md").href = `/api/lectures/${lecture.id}/export.md`;
  $("#dl-csv").href = `/api/lectures/${lecture.id}/flashcards.csv`;
  $("#dl-vtt").href = `/api/lectures/${lecture.id}/transcript.vtt`;
  $("#dl-vtt").hidden = !lecture.has_timestamps;
  $("#recent").hidden = true;

  const warnings = $("#warnings");
  warnings.replaceChildren(...pack.warnings.map((w) => h("li", {}, w)));
  warnings.hidden = pack.warnings.length === 0;

  $("#outline").replaceChildren(...pack.sections.map((s) =>
    h("li", {},
      h("div", {}, h("span", { class: "heading" }, s.heading), tsButton(s.chunk_ids[0], s.start)),
      h("div", { class: "muted" }, s.summary))));

  $("#points").replaceChildren(...pack.key_points.map((p) =>
    h("li", {}, p.text, tsButton(p.chunk_id, itemStart(p)))));

  $("#glossary").replaceChildren(...pack.terms.flatMap((t) => [
    h("dt", {}, t.term, tsButton(t.chunk_id, itemStart(t))),
    h("dd", {}, t.definition),
  ]));
  if (!pack.terms.length) $("#glossary").replaceChildren(h("p", { class: "muted" }, "No glossary terms found."));

  state.cards = pack.flashcards.slice();
  state.cardIndex = 0;
  state.flipped = false;
  renderCard();
  renderQuiz();
  $("#answers").replaceChildren();
  renderTranscript();
  selectTab("outline");
}

// ---------- tabs ----------

function selectTab(name) {
  for (const tab of document.querySelectorAll("[role=tab]")) {
    tab.setAttribute("aria-selected", String(tab.dataset.tab === name));
  }
  for (const panel of document.querySelectorAll(".panel")) {
    panel.hidden = panel.dataset.panel !== name;
  }
}
for (const tab of document.querySelectorAll("[role=tab]")) {
  tab.addEventListener("click", () => selectTab(tab.dataset.tab));
}

// ---------- flashcards ----------

function renderCard() {
  const empty = state.cards.length === 0;
  $("#card-empty").hidden = !empty;
  $("#card-wrap").hidden = empty;
  if (empty) return;
  const card = state.cards[state.cardIndex];
  const el = $("#flashcard");
  el.classList.toggle("back", state.flipped);
  $("#card-face").textContent = state.flipped ? "Answer" : "Question";
  $("#card-text").textContent = state.flipped ? card.back : card.front;
  $("#card-count").textContent = `${state.cardIndex + 1} / ${state.cards.length}`;
}

function moveCard(step) {
  if (!state.cards.length) return;
  state.cardIndex = (state.cardIndex + step + state.cards.length) % state.cards.length;
  state.flipped = false;
  renderCard();
}

$("#flashcard").addEventListener("click", () => { state.flipped = !state.flipped; renderCard(); });
$("#card-prev").addEventListener("click", () => moveCard(-1));
$("#card-next").addEventListener("click", () => moveCard(1));
$("#card-shuffle").addEventListener("click", () => {
  for (let i = state.cards.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [state.cards[i], state.cards[j]] = [state.cards[j], state.cards[i]];
  }
  state.cardIndex = 0;
  state.flipped = false;
  renderCard();
});
document.addEventListener("keydown", (e) => {
  const onCards = !document.querySelector('[data-panel="cards"]').hidden;
  if (!onCards || ["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
  if (e.key === "ArrowRight") moveCard(1);
  else if (e.key === "ArrowLeft") moveCard(-1);
  else if (e.key === " " && document.activeElement !== $("#flashcard")) {
    e.preventDefault();
    state.flipped = !state.flipped;
    renderCard();
  }
});

// ---------- quiz ----------

function renderQuiz() {
  const quiz = state.lecture.pack.quiz;
  $("#quiz-score").textContent = "";
  $("#quiz-check").hidden = $("#quiz-reset").hidden = quiz.length === 0;
  if (!quiz.length) {
    $("#quiz").replaceChildren(h("p", { class: "muted" }, "No quiz questions for this lecture."));
    return;
  }
  $("#quiz").replaceChildren(...quiz.map((q, qi) =>
    h("div", { class: "question" },
      h("fieldset", {},
        h("legend", {}, `${qi + 1}. ${q.question}`),
        ...q.options.map((opt, oi) =>
          h("label", { class: "option" },
            h("input", { type: "radio", name: `q${qi}`, value: String(oi) }),
            h("span", {}, opt))),
        h("p", { class: "explanation", hidden: true },
          q.explanation, tsButton(q.chunk_id, itemStart(q)))))));
}

$("#quiz-check").addEventListener("click", () => {
  const quiz = state.lecture.pack.quiz;
  let correct = 0;
  quiz.forEach((q, qi) => {
    const block = $("#quiz").children[qi];
    const chosen = block.querySelector(`input[name="q${qi}"]:checked`);
    block.querySelectorAll(".option").forEach((label, oi) => {
      label.classList.toggle("correct", oi === q.answer);
      label.classList.toggle("wrong", chosen != null && Number(chosen.value) === oi && oi !== q.answer);
    });
    block.querySelector(".explanation").hidden = false;
    if (chosen && Number(chosen.value) === q.answer) correct += 1;
  });
  $("#quiz-score").textContent = `Score: ${correct} / ${quiz.length}`;
});
$("#quiz-reset").addEventListener("click", renderQuiz);

// ---------- ask ----------

$("#ask-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = $("#question");
  const question = input.value.trim();
  if (!question) return;
  input.value = "";
  const pending = h("div", { class: "qa" }, h("div", { class: "q" }, question),
    h("div", { class: "thinking" }, "Thinking…"));
  $("#answers").prepend(pending);
  try {
    const answer = await api(`/api/lectures/${state.lecture.id}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    pending.replaceWith(h("div", { class: `qa${answer.grounded ? "" : " ungrounded"}` },
      h("div", { class: "q" }, answer.question),
      h("div", { class: "a" }, answer.answer),
      h("div", {}, answer.citations.map((c) => tsButton(c.chunk_id, c.start)))));
  } catch (err) {
    pending.querySelector(".thinking").textContent = err.message;
  }
});

// ---------- transcript ----------

function renderTranscript() {
  $("#transcript").replaceChildren(...state.lecture.chunks.map((c) =>
    h("div", { class: "chunk", id: `chunk-${c.id}` },
      h("div", { class: "chunk-head" },
        c.start != null ? `Part ${c.id} · ${fmtTime(c.start)}–${fmtTime(c.end)}` : `Part ${c.id}`),
      h("div", {}, c.text))));
}

function jumpTo(chunkId) {
  selectTab("transcript");
  const el = document.getElementById(`chunk-${chunkId}`);
  if (!el) return;
  el.scrollIntoView({ behavior: "smooth", block: "center" });
  el.classList.add("flash");
  setTimeout(() => el.classList.remove("flash"), 1600);
}

// ---------- recent lectures ----------

async function loadRecent() {
  let items = [];
  try { items = await api("/api/lectures"); } catch { /* list is optional */ }
  const section = $("#recent");
  section.hidden = items.length === 0 || !$("#results").hidden;
  $("#recent-list").replaceChildren(...items.map((item) => {
    const when = new Date(item.created_at).toLocaleString();
    const row = h("li", {},
      h("div", {},
        h("button", { class: "open", type: "button", onclick: () => openLecture(item.id) }, item.title),
        h("div", { class: "meta" }, `${when} · ${item.word_count.toLocaleString()} words · ${item.engine}`)),
      h("button", {
        class: "delete", type: "button", title: `Delete “${item.title}”`,
        onclick: async () => {
          if (!confirm(`Delete “${item.title}”?`)) return;
          try {
            await fetch(`/api/lectures/${encodeURIComponent(item.id)}`, { method: "DELETE" });
          } finally {
            loadRecent();
          }
        },
      }, "Delete"));
    return row;
  }));
}

async function openLecture(id) {
  try { show(await api(`/api/lectures/${encodeURIComponent(id)}`)); } catch (err) { status(err.message); }
}

// ---------- boot ----------

(async () => {
  try {
    const health = await api("/api/health");
    $("#engine").textContent = health.model ? `${health.engine} · ${health.model}` : `${health.engine} engine`;
    state.audio = health.audio;
    if (health.audio) {
      $("#file").accept += ",audio/*,.mp3,.m4a,.wav,.webm,.ogg,.oga,.flac,.mp4,.mpeg,.mpga";
      state.dropLabel = "Drop a transcript or recording here, or click to browse";
      $("#drop-label").textContent = state.dropLabel;
      $("#audio-hint").hidden = false;
    }
    $("#recent-note").textContent = health.persistent
      ? "Saved on this server."
      : "Kept in memory; restarting the server clears them (use --db to keep them).";
  } catch {
    $("#engine").textContent = "server unreachable";
  }
  const id = location.hash.slice(1);
  if (id) {
    try { show(await api(`/api/lectures/${encodeURIComponent(id)}`)); return; } catch { /* gone */ }
  }
  loadRecent();
})();
