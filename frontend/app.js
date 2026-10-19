"use strict";

const HISTORY_SENT = 20;           // messages sent as chat_context
const MAX_MESSAGE_CHARS = 20000;   // server limit per message
const ANSWER_TIMEOUT_MS = 120000;
const CHATS_LIMIT = 50;
const NEW_CHAT_TITLE = "New chat";
const MAX_TITLE_CHARS = 60;
const READY_POLL_MS = 2000;
const WAITING_TEXT = "Processing your request…";
// Error codes after which the server closes the socket.
const CLOSING_CODES = new Set(["MAX_CONNECTIONS_REACHED", "IDLE_TIMEOUT"]);

const el = (id) => document.getElementById(id);
const ui = {
  uploadForm: el("upload-form"),
  files: el("files"),
  uploadButton: el("upload-button"),
  uploadLimits: el("upload-limits"),
  uploadResults: el("upload-results"),
  statusDot: el("status-dot"),
  statusText: el("status-text"),
  language: el("language"),
  newChat: el("new-chat"),
  chatList: el("chat-list"),
  messages: el("messages"),
  chatForm: el("chat-form"),
  question: el("question"),
  sendButton: el("send-button"),
};

const state = {
  config: { index_name: "Documents", namespace: "default", file_types: [], max_file_size_mb: 0, max_question_chars: 4000, persistent_chats: false },
  socket: null,
  socketKey: null,
  threadId: null,
  chats: [],           // [{id, title, updated}], newest first
  history: [],         // [{type: "human"|"ai", content}]
  pending: null,       // answer being streamed
  ready: false,        // true once /ready returns 200
};

// Storage may be unavailable; the page still works.
function load(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw === null ? fallback : JSON.parse(raw);
  } catch {
    return fallback;
  }
}

function save(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* ignore */
  }
}

// The server fixes the index/namespace.
function target() {
  const { index_name: index, namespace } = state.config;
  return { index, namespace, key: `${index}/${namespace}` };
}

const historyKey = (id = state.threadId) => `rag.history.${id}`;

function newId() {
  if (crypto.randomUUID) return crypto.randomUUID();
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = [...bytes].map((b) => b.toString(16).padStart(2, "0")).join("");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

function saveChats() {
  save("rag.chats", state.chats);
  save("rag.activeChat", state.threadId);
}

let editingId = null;   // chat being renamed

// A generated title only applies while the chat is still auto-named (not renamed by the user).
function setChatTitle(id, title, fromModel = false) {
  const chat = state.chats.find((c) => c.id === id);
  if (!chat || (fromModel && (!chat.auto || editingId === id))) return;
  chat.title = title;
  chat.auto = false;
  saveChats();
  renderChats();
}

function startRename(chat, item) {
  editingId = chat.id;
  const input = document.createElement("input");
  input.type = "text";
  input.className = "chat-rename";
  input.value = chat.title;
  input.maxLength = MAX_TITLE_CHARS;
  input.setAttribute("aria-label", "Chat name");

  let finished = false;
  const finish = (commit) => {
    if (finished) return;
    finished = true;
    editingId = null;
    const title = input.value.trim();
    if (commit && title && title !== chat.title) setChatTitle(chat.id, title);
    else renderChats();
  };
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") finish(true);
    else if (event.key === "Escape") finish(false);
  });
  input.addEventListener("blur", () => finish(true));

  item.replaceChildren(input);
  input.focus();
  input.select();
}

// Runs in parallel with the answer; on any failure the first-words title stays.
async function requestTitle(id, question) {
  try {
    const response = await fetch("/chat/title", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!response.ok) return;
    const { title } = await response.json();
    if (title) setChatTitle(id, title, true);
  } catch {
    /* keep the fallback title */
  }
}

function renderChats() {
  ui.chatList.replaceChildren(
    ...state.chats.map((chat) => {
      const item = document.createElement("li");
      item.classList.toggle("active", chat.id === state.threadId);

      const open = document.createElement("button");
      open.type = "button";
      open.className = "chat-item";
      open.textContent = chat.title;
      open.title = chat.title;
      open.addEventListener("click", () => openChat(chat.id));
      open.addEventListener("dblclick", () => startRename(chat, item));

      const rename = document.createElement("button");
      rename.type = "button";
      rename.className = "chat-edit";
      rename.textContent = "✎";
      rename.setAttribute("aria-label", `Rename chat: ${chat.title}`);
      rename.addEventListener("click", () => startRename(chat, item));

      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "chat-delete";
      remove.textContent = "×";
      remove.setAttribute("aria-label", `Remove chat: ${chat.title}`);
      remove.addEventListener("click", () => removeChat(chat.id));

      item.append(open, rename, remove);
      return item;
    }),
  );
}

// Add the open chat to the sidebar and move it to the top.
function touchChat(question) {
  let chat = state.chats.find((c) => c.id === state.threadId);
  if (!chat) {
    chat = { id: state.threadId, title: NEW_CHAT_TITLE, updated: 0, auto: true };
    state.chats.push(chat);
  }
  if (chat.title === NEW_CHAT_TITLE) chat.title = question.length > 40 ? `${question.slice(0, 40)}…` : question;
  chat.updated = Date.now();
  state.chats.sort((a, b) => b.updated - a.updated);
  state.chats = state.chats.slice(0, CHATS_LIMIT);
  saveChats();
  renderChats();
}

async function loadThreadHistory(id) {
  if (!state.config.persistent_chats) return load(historyKey(id), []);
  try {
    const response = await fetch(`/chat/threads/${encodeURIComponent(id)}/messages`, { cache: "no-store" });
    return response.ok ? (await response.json()).messages : [];
  } catch {
    return [];
  }
}

async function openChat(id) {
  if (state.pending || id === state.threadId) return;
  closeSocket();
  state.threadId = id;
  saveChats();
  renderChats();
  state.history = [];
  renderHistory();
  const history = await loadThreadHistory(id);
  if (state.threadId !== id) return;
  state.history = history;
  renderHistory();
  ui.question.focus();
}

function removeChat(id) {
  if (state.pending) return;
  state.chats = state.chats.filter((c) => c.id !== id);
  try {
    localStorage.removeItem(historyKey(id));
  } catch {
    /* ignore */
  }
  if (id === state.threadId) {
    state.threadId = null;
    if (state.chats.length) openChat(state.chats[0].id);
    else newChat();
  }
  saveChats();
  renderChats();
}

function setStatus(kind, text) {
  ui.statusDot.className = `dot ${kind}`;
  ui.statusText.textContent = text;
}

// Markdown rendering: text is HTML-escaped first, then only fixed tags are added.
function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function renderInline(escaped) {
  return escaped
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*([^*\s][^*]*?)\*(?!\w)/g, "$1<em>$2</em>");
}

// Math: \( ... \), \[ ... \] and $$ ... $$ are rendered with KaTeX; if it has not loaded they stay as text.
// KaTeX output is kept aside and put back after escaping, so the escaped text is never mixed with it.
const MATH_PATTERN = /\\\[([\s\S]+?)\\\]|\\\(([\s\S]+?)\\\)|\$\$([\s\S]+?)\$\$/g;

function extractMath(text, store) {
  if (typeof katex === "undefined") return text;
  return text.replace(MATH_PATTERN, (match, block, inline, dollars) => {
    try {
      const tex = (block ?? inline ?? dollars).trim();
      store.push(katex.renderToString(tex, { displayMode: inline === undefined, throwOnError: false, trust: false }));
    } catch {
      return match;
    }
    return `\u0000${store.length - 1}\u0000`;
  });
}

function renderMarkdown(text) {
  const math = [];
  const lines = escapeHtml(extractMath(text, math)).split("\n");
  const html = [];
  let paragraph = [];
  let list = null;        // "ul" or "ol"
  let code = null;        // lines of the open code block

  const flushParagraph = () => {
    if (paragraph.length) html.push(`<p>${paragraph.map(renderInline).join("<br>")}</p>`);
    paragraph = [];
  };
  const closeList = () => {
    if (list) html.push(`</${list}>`);
    list = null;
  };

  for (const line of lines) {
    if (code !== null) {
      if (line.trim().startsWith("```")) {
        html.push(`<pre><code>${code.join("\n")}</code></pre>`);
        code = null;
      } else {
        code.push(line);
      }
      continue;
    }
    if (line.trim().startsWith("```")) {
      flushParagraph(); closeList(); code = [];
      continue;
    }

    const heading = line.match(/^(#{1,4})\s+(.*)$/);
    const bullet = line.match(/^\s*[-*•]\s+(.*)$/);
    const numbered = line.match(/^\s*\d+[.)]\s+(.*)$/);

    if (heading) {
      flushParagraph(); closeList();
      html.push(`<h4>${renderInline(heading[2])}</h4>`);
    } else if (bullet || numbered) {
      flushParagraph();
      const kind = bullet ? "ul" : "ol";
      if (list !== kind) { closeList(); html.push(`<${kind}>`); list = kind; }
      html.push(`<li>${renderInline((bullet || numbered)[1])}</li>`);
    } else if (line.trim() === "") {
      // A blank line does not end a list: items separated by blank lines stay in one list, so numbering continues.
      flushParagraph();
    } else {
      closeList();
      paragraph.push(line);
    }
  }
  if (code !== null) html.push(`<pre><code>${code.join("\n")}</code></pre>`);
  flushParagraph(); closeList();
  return html.join("").replace(/\u0000(\d+)\u0000/g, (_match, index) => math[index]);
}

function setBubbleText(bubble, text) {
  const isAnswer = bubble.classList.contains("ai")
    && !bubble.classList.contains("error")
    && !bubble.classList.contains("waiting");
  if (isAnswer) {
    bubble.innerHTML = renderMarkdown(text);
  } else {
    bubble.textContent = text;
  }
}

function addBubble(kind, text = "") {
  ui.messages.querySelector(".empty")?.remove();
  const bubble = document.createElement("div");
  bubble.className = `message ${kind}`;
  setBubbleText(bubble, text);
  ui.messages.append(bubble);
  scrollToBottom();
  return bubble;
}

function scrollToBottom() {
  ui.messages.scrollTop = ui.messages.scrollHeight;
}

function renderHistory() {
  ui.messages.replaceChildren();
  if (state.history.length === 0) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "Ask a question about your documents.";
    ui.messages.append(empty);
    return;
  }
  for (const message of state.history) addBubble(message.type, message.content);
}

function socketUrl({ index, namespace }) {
  const scheme = location.protocol === "https:" ? "wss:" : "ws:";
  const path = `/ws/chat/${encodeURIComponent(index)}/${encodeURIComponent(namespace)}`;
  return `${scheme}//${location.host}${path}?thread_id=${encodeURIComponent(state.threadId)}`;
}

function closeSocket() {
  if (state.socket) {
    const old = state.socket;
    state.socket = null;
    state.socketKey = null;
    old.close();
  }
  if (state.ready) setStatus("", "Not connected");
}

// Open or reuse the socket; it connects only when a question is sent.
function openSocket() {
  const t = target();
  const socketKey = `${t.key}#${state.threadId}`;
  if (state.socket && state.socketKey === socketKey && state.socket.readyState === WebSocket.OPEN) {
    return Promise.resolve(state.socket);
  }
  closeSocket();
  setStatus("connecting", "Connecting…");

  return new Promise((resolve, reject) => {
    const socket = new WebSocket(socketUrl(t));
    state.socket = socket;
    state.socketKey = socketKey;

    socket.onopen = () => {
      setStatus("connected", "Connected");
      resolve(socket);
    };
    socket.onmessage = onFrame;
    socket.onerror = () => reject(new Error("Could not connect to the server."));
    socket.onclose = () => {
      if (state.socket === socket) {
        state.socket = null;
        state.socketKey = null;
        // Keep a refusal message visible.
        if (!ui.statusDot.classList.contains("error")) setStatus("", "Not connected");
      }
      if (state.pending) finishAnswer("The connection closed before the answer finished. Please retry.");
    };
  });
}

function onFrame(event) {
  if (event.data === "pong") return;
  let frame;
  try {
    frame = JSON.parse(event.data);
  } catch {
    return;
  }

  if (frame.type === "stream" && state.pending) {
    state.pending.bubble.classList.remove("waiting");
    state.pending.text += frame.message;
    setBubbleText(state.pending.bubble, state.pending.text);
    scrollToBottom();
  } else if (frame.type === "end" && state.pending) {
    finishAnswer(null);
  } else if (frame.type === "error") {
    if (CLOSING_CODES.has(frame.code)) setStatus("error", frame.message);
    if (state.pending) {
      finishAnswer(friendlyError(frame));
    } else if (frame.code !== "IDLE_TIMEOUT") {
      addBubble("system", friendlyError(frame));
    }
  }
}

function friendlyError(frame) {
  return frame.message || "Something went wrong. Please retry.";
}

async function ask(question) {
  if (state.pending) return;

  const chatContext = state.history.slice(-HISTORY_SENT).map((m) => ({
    type: m.type,
    content: m.content.slice(0, MAX_MESSAGE_CHARS),
  }));

  const isNewChat = !state.chats.some((c) => c.id === state.threadId);
  touchChat(question);
  addBubble("human", question);
  const bubble = addBubble("ai pending waiting", WAITING_TEXT);
  state.pending = {
    question,
    text: "",
    bubble,
    timer: setTimeout(() => finishAnswer("The answer took too long. Please retry."), ANSWER_TIMEOUT_MS),
  };
  setBusy(true);

  try {
    const socketReady = openSocket();
    if (isNewChat) requestTitle(state.threadId, question);   // runs alongside the answer, never awaited
    const socket = await socketReady;
    socket.send(JSON.stringify({ question, language: ui.language.value, chat_context: chatContext }));
  } catch (error) {
    setStatus("error", "Not connected");
    finishAnswer(error.message);
  }
}

// error is null when the answer finished normally.
function finishAnswer(error) {
  const pending = state.pending;
  if (!pending) return;
  state.pending = null;
  clearTimeout(pending.timer);
  pending.bubble.classList.remove("pending", "waiting");

  if (error) {
    // Show partial text, but keep it out of the history.
    pending.bubble.classList.add("error");
    pending.bubble.textContent = pending.text ? `${pending.text}\n\n${error}` : error;
  } else if (pending.text) {
    state.history.push(
      { type: "human", content: pending.question },
      { type: "ai", content: pending.text },
    );
    // Without server-side memory, keep the history in the browser.
    if (!state.config.persistent_chats) save(historyKey(), state.history.slice(-HISTORY_SENT * 2));
  }
  setBusy(false);
  ui.question.focus();
}

function setBusy(busy) {
  ui.sendButton.disabled = busy || !state.ready;
  ui.sendButton.textContent = busy ? "Answering…" : "Send";
}

// Lock the page until /ready returns 200.
function setLocked(locked) {
  state.ready = !locked;
  ui.question.disabled = locked;
  ui.sendButton.disabled = locked || Boolean(state.pending);
  ui.uploadButton.disabled = locked;
  ui.question.placeholder = locked ? "Warming up, please wait…" : "Ask about your documents…";
}

async function serverIsReady() {
  try {
    const response = await fetch("/ready", { cache: "no-store" });
    return response.ok;
  } catch {
    return false;
  }
}

async function waitUntilReady() {
  setLocked(true);
  setStatus("connecting", "Warming up… you can ask in a moment");
  while (!(await serverIsReady())) {
    await new Promise((resolve) => setTimeout(resolve, READY_POLL_MS));
  }
  setLocked(false);
  setStatus("", "Not connected");
  ui.question.focus();
}

function newChat() {
  if (state.pending) return;
  // An empty chat is already new.
  if (state.threadId && state.history.length === 0) {
    ui.question.focus();
    return;
  }
  closeSocket();
  state.threadId = newId();
  state.history = [];
  saveChats();
  renderChats();
  renderHistory();
  ui.question.focus();
}

function fileProblem(file) {
  const type = file.name.split(".").pop().toLowerCase();
  const { file_types: types, max_file_size_mb: maxMb } = state.config;
  if (types.length && !types.includes(type)) return `type .${type} is not supported`;
  if (maxMb && file.size > maxMb * 1024 * 1024) return `larger than ${maxMb} MB`;
  return null;
}

function addResult(kind, text) {
  const item = document.createElement("li");
  item.className = kind;
  item.textContent = text;
  ui.uploadResults.append(item);
  return item;
}

async function upload(event) {
  event.preventDefault();
  const files = [...ui.files.files];
  if (files.length === 0) return;

  ui.uploadResults.replaceChildren();
  const accepted = [];
  for (const file of files) {
    const problem = fileProblem(file);
    if (problem) addResult("failed", `${file.name}: not uploaded, ${problem}`);
    else accepted.push(file);
  }
  if (accepted.length === 0) return;

  const form = new FormData();
  for (const file of accepted) form.append("files", file);

  ui.uploadButton.disabled = true;
  const busy = addResult("busy", `Uploading ${accepted.length} file(s)… large files take a while.`);
  try {
    const response = await fetch("/ingest", { method: "POST", body: form });
    const body = await response.json().catch(() => null);
    busy.remove();
    if (!response.ok) {
      addResult("failed", `Upload failed: ${body?.detail ?? response.statusText}`);
      return;
    }
    for (const result of body) {
      if (result.status === "COMPLETED") {
        const pages = result.total_pages ? ` (${result.total_pages} pages)` : "";
        addResult("ok", `${result.file_name}: ready${pages}`);
      } else {
        addResult("failed", `${result.file_name}: ${result.error_detail || "failed"}`);
      }
    }
    ui.uploadForm.reset();
  } catch {
    busy.remove();
    addResult("failed", "Upload failed: the server could not be reached.");
  } finally {
    ui.uploadButton.disabled = false;
  }
}

async function loadConfig() {
  try {
    const response = await fetch("/ui-config");
    if (response.ok) state.config = { ...state.config, ...(await response.json()) };
  } catch {
    /* keep defaults */
  }
}

function autoGrow() {
  ui.question.style.height = "auto";
  ui.question.style.height = `${ui.question.scrollHeight}px`;
}

async function start() {
  await loadConfig();
  const { config } = state;

  ui.question.maxLength = config.max_question_chars;
  if (config.file_types.length) {
    ui.files.accept = config.file_types.map((type) => `.${type}`).join(",");
    ui.uploadLimits.textContent =
      `Allowed: ${config.file_types.join(", ")}` + (config.max_file_size_mb ? ` · up to ${config.max_file_size_mb} MB each` : "");
  }

  state.chats = load("rag.chats", []);
  state.threadId = load("rag.activeChat", null) || newId();
  state.history = await loadThreadHistory(state.threadId);
  renderChats();
  renderHistory();

  ui.uploadForm.addEventListener("submit", upload);
  ui.newChat.addEventListener("click", newChat);
  ui.question.addEventListener("input", autoGrow);
  ui.question.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      ui.chatForm.requestSubmit();
    }
  });
  ui.chatForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const question = ui.question.value.trim();
    if (!question || state.pending || !state.ready) return;
    ui.question.value = "";
    autoGrow();
    ask(question);
  });
  waitUntilReady();
}

start();