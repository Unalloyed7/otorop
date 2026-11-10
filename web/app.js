"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const shell = document.querySelector(".app-shell");
  const storageKey = "psb-legal-conversations-v1";
  const statusLabels = {
    answer: "Ответ с основаниями", partial: "Подтверждена часть", clarify: "Нужно уточнение",
    insufficient: "Недостаточно оснований", out_of_scope: "Вне тематики", refuse: "Отказ", error: "Техническая ошибка",
  };
  const typeLabels = { agreement: "Договор", amendment: "Допсоглашение", internal_policy: "Регламент", letter: "Письмо / записка", law: "Законодательство", other: "Документ" };
  let conversations = [];
  let activeId = null;
  let busy = false;
  let serviceReady = false;
  let documents = [];
  let documentsLoaded = false;
  let documentToken = 0;
  let quoteMark = null;
  let panelFocus = null;
  let toastTimer;

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function toast(text) {
    $("toast").textContent = text;
    $("toast").hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { $("toast").hidden = true; }, 3500);
  }
  function current() { return conversations.find((c) => c.id === activeId); }
  function persist() {
    try { sessionStorage.setItem(storageKey, JSON.stringify({ activeId, conversations })); }
    catch { /* Память вкладки остаётся доступной и при ограничении storage. */ }
  }
  function restore() {
    try {
      const data = JSON.parse(sessionStorage.getItem(storageKey) || "null");
      if (data && Array.isArray(data.conversations)) {
        conversations = data.conversations.filter((c) => typeof c.id === "string" && typeof c.title === "string" && Array.isArray(c.messages) && c.messages.length <= 40 && c.messages.every((m) => ["user", "assistant"].includes(m.role) && typeof m.content === "string")).slice(0, 15);
        activeId = conversations.some((c) => c.id === data.activeId) ? data.activeId : conversations[0]?.id;
      }
    } catch { conversations = []; }
    if (!activeId) newConversation();
  }
  function newConversation() {
    if (busy) return;
    const empty = conversations.find((c) => c.messages.length === 0);
    if (empty) activeId = empty.id;
    else {
      const id = crypto.randomUUID?.() || `${Date.now()}-${Math.random().toString(36).slice(2)}`;
      const c = { id, title: "Новый диалог", messages: [], asOf: "" };
      conversations.unshift(c);
      conversations = conversations.slice(0, 15);
      activeId = c.id;
    }
    $("question").value = "";
    $("question").style.height = "auto";
    closeSidebar();
    render();
    persist();
  }
  function renderSidebar() {
    const list = $("conversation-list");
    list.replaceChildren();
    for (const c of conversations) {
      const button = element("button", "conversation-button" + (c.id === activeId ? " active" : ""));
      button.type = "button";
      button.append(element("span", "conversation-symbol", "▱"), element("span", "", c.title));
      button.title = c.title;
      if (c.id === activeId) button.setAttribute("aria-current", "true");
      button.addEventListener("click", () => {
        if (busy) return;
        activeId = c.id;
        closeSidebar();
        render();
        persist();
      });
      list.append(button);
    }
  }
  function formatText(node, text, citations) {
    // Никакого HTML/Markdown из ответа или исходника. Только текст и наши кнопки.
    const regex = /\[(\d+)\]/g;
    let start = 0;
    for (const match of text.matchAll(regex)) {
      node.append(document.createTextNode(text.slice(start, match.index)));
      const citation = citations.find((c) => c.claim_index === Number(match[1]) - 1);
      if (citation) {
        const button = element("button", "inline-citation", match[0]);
        button.type = "button";
        button.setAttribute("aria-label", `Открыть основание ${match[1]}`);
        button.addEventListener("click", () => openCitation(citation));
        node.append(button);
      } else node.append(document.createTextNode(match[0]));
      start = match.index + match[0].length;
    }
    node.append(document.createTextNode(text.slice(start)));
  }
  function renderMessage(message, index) {
    const node = element("article", "message message-" + message.role);
    if (message.role === "user") {
      node.append(element("div", "message-text", message.content));
      return node;
    }
    const answer = message.answer || { status: "error", text: message.content, citations: [] };
    const label = element("div", "message-label");
    label.append(element("span", "assistant-avatar", "▤"), document.createTextNode("АССИСТЕНТ ПСБ"), element("span", "message-status " + answer.status, statusLabels[answer.status] || "Ответ"));
    const text = element("div", "message-text");
    const citations = Array.isArray(answer.citations) ? answer.citations : [];
    formatText(text, message.content, citations);
    node.append(label, text);
    if (citations.length) {
      node.append(element("div", "evidence-heading", `ОСНОВАНИЯ · ${citations.length}`));
      const cards = element("div", "citations");
      for (const c of citations) {
        const button = element("button", "citation-card");
        button.type = "button";
        const body = element("span", "citation-body");
        body.append(element("span", "citation-title", c.document_title), element("span", "citation-meta", c.section + " · " + c.edition), element("span", "citation-quote", "«" + c.quote + "»"));
        button.append(element("span", "citation-number", String(c.claim_index + 1).padStart(2, "0")), body, element("span", "citation-arrow", "↗"));
        button.addEventListener("click", () => openCitation(c));
        cards.append(button);
      }
      node.append(cards);
    }
    const actions = element("div", "message-actions");
    const copy = element("button", "text-button", "Скопировать ответ");
    copy.type = "button";
    copy.addEventListener("click", async () => {
      const sources = citations.map((c) => `${c.document_title}, ${c.section} (${c.edition}). «${c.quote}»`).join("\n");
      try { await navigator.clipboard.writeText(message.content + (sources ? "\n\nОснования:\n" + sources : "")); toast("Ответ скопирован вместе с основаниями"); }
      catch { toast("Не удалось скопировать. Выделите текст ответа вручную."); }
    });
    actions.append(copy);
    if (answer.status === "error" && index === current().messages.length - 1) {
      const retry = element("button", "text-button retry-button", "Повторить запрос");
      retry.type = "button";
      retry.addEventListener("click", () => sendQuestion(true));
      actions.append(retry);
    }
    node.append(actions);
    return node;
  }
  function render() {
    renderSidebar();
    const c = current();
    $("as-of").value = c.asOf || "";
    $("welcome").hidden = c.messages.length > 0;
    $("messages").hidden = c.messages.length === 0;
    $("messages").replaceChildren(...c.messages.map(renderMessage));
    updateControls();
  }
  function updateControls() {
    $("new-chat").disabled = busy;
    $("send-button").disabled = busy || !serviceReady || !$("question").value.trim();
    $("question").disabled = busy;
    $("as-of").disabled = busy;
    $("chat-form").setAttribute("aria-busy", String(busy));
    for (const button of document.querySelectorAll(".suggestion")) button.disabled = busy;
  }
  function scrollToEnd() { $("chat-area").scrollTop = $("chat-area").scrollHeight; }
  async function request(url, options = {}) {
    const response = await fetch(url, { ...options, headers: { ...options.headers }, cache: "no-store" });
    if (!response.ok) {
      if (response.status === 429) throw new Error("Сервис занят. Повторите запрос немного позже.");
      if (response.status === 404) throw new Error("Документ или фрагмент больше не найден в текущей базе.");
      if (response.status === 422 || response.status === 413) throw new Error("Запрос слишком длинный или его формат не поддерживается. Начните новый диалог.");
      throw new Error("Не удалось связаться с сервисом. Проверьте подключение и повторите запрос.");
    }
    return response.json();
  }
  async function sendQuestion(retry = false) {
    if (busy || !serviceReady) return;
    const c = current();
    if (retry) {
      const lastUser = c.messages.map((m) => m.role).lastIndexOf("user");
      if (lastUser < 0) return;
      c.messages = c.messages.slice(0, lastUser + 1);
    } else {
      const question = $("question").value.trim();
      if (!question) return;
      if (c.messages.length >= 39 || c.messages.reduce((n, m) => n + m.content.length, 0) + question.length > 58000) {
        toast("Диалог стал слишком длинным. Начните новый диалог.");
        return;
      }
      c.messages.push({ role: "user", content: question });
      if (c.messages.length === 1) c.title = question.length > 55 ? question.slice(0, 54) + "…" : question;
      c.asOf = $("as-of").value;
      $("question").value = "";
      $("question").style.height = "auto";
    }
    busy = true;
    render();
    const loading = element("div", "loading-message");
    const dots = element("span", "loading-dots");
    for (let i = 0; i < 3; i++) dots.append(element("i"));
    loading.append(dots, document.createTextNode("Проверяю вопрос по базе документов…"));
    loading.setAttribute("role", "status");
    $("messages").append(loading);
    scrollToEnd();
    let answer;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 275000);
    try {
      answer = await request("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, signal: controller.signal, body: JSON.stringify({ messages: c.messages.map(({ role, content }) => ({ role, content })), as_of: c.asOf || null }) });
      if (!answer || typeof answer.text !== "string" || !statusLabels[answer.status]) throw new Error("Сервис вернул ответ в неизвестном формате. Повторите запрос.");
    } catch (error) {
      answer = { status: "error", text: error.name === "AbortError" ? "Истекло время ожидания ответа. Попробуйте повторить запрос." : error.message, citations: [], missing_info: [] };
    } finally { clearTimeout(timeout); }
    c.messages.push({ role: "assistant", content: answer.text, answer });
    busy = false;
    persist();
    render();
    scrollToEnd();
    $("question").focus({ preventScroll: true });
  }
  async function refreshStatus() {
    try {
      const status = await request("/api/status");
      serviceReady = status.ready;
      $("service-status").classList.toggle("unavailable", !status.ready);
      $("service-status-text").textContent = status.ready ? "Готов к работе" : (status.documents_available ? "Документы доступны" : "Требуется подключение");
      $("document-count").textContent = status.documents_available ? `${status.document_count} в базе знаний` : "База пока недоступна";
      $("service-notice").hidden = status.ready;
      $("service-notice").textContent = !status.documents_available ? "База документов пока недоступна. После её подключения можно будет просматривать источники и задавать вопросы." : "Модель пока не подключена. Вы можете открыть и изучить документы базы знаний.";
    } catch {
      serviceReady = false;
      $("service-status").classList.add("unavailable");
      $("service-status-text").textContent = "Нет соединения";
      $("service-notice").hidden = false;
      $("service-notice").textContent = "Не удалось подключиться к сервису. Обновите страницу немного позже.";
    }
    updateControls();
  }
  function updatePanelMode() {
    const modal = !$("document-panel").hidden && window.innerWidth <= 1000;
    if (modal) {
      $("document-panel").setAttribute("role", "dialog");
      $("document-panel").setAttribute("aria-modal", "true");
    } else {
      $("document-panel").removeAttribute("role");
      $("document-panel").removeAttribute("aria-modal");
    }
    $("main-area").inert = modal || $("sidebar").classList.contains("open");
    $("sidebar").inert = modal;
  }
  function setPanel(open) {
    if (open) { closeSidebar(); panelFocus = document.activeElement; }
    $("document-panel").hidden = !open;
    shell.classList.toggle("panel-open", open);
    $("sources-toggle").setAttribute("aria-expanded", String(open));
    updatePanelMode();
    if (open) $("close-panel").focus();
    else if (panelFocus && document.contains(panelFocus) && panelFocus.getClientRects().length) panelFocus.focus({ preventScroll: true });
    else $("sources-toggle").focus({ preventScroll: true });
  }
  async function openLibrary() {
    setPanel(true);
    documentToken++;
    $("document-library").hidden = false;
    $("document-view").hidden = true;
    $("panel-eyebrow").textContent = "БАЗА ЗНАНИЙ";
    if (documentsLoaded) return renderDocuments();
    $("document-list").replaceChildren(element("p", "empty-list", "Загружаем документы…"));
    try {
      const result = await request("/api/documents");
      documents = result.documents;
      documentsLoaded = true;
      renderDocuments();
    } catch (error) { $("document-list").replaceChildren(element("p", "empty-list", error.message)); }
  }
  function renderDocuments() {
    const query = $("document-query").value.toLocaleLowerCase("ru");
    const rows = documents.filter((d) => [d.title, d.source_file, d.contract_number, d.counterparty].filter(Boolean).join(" ").toLocaleLowerCase("ru").includes(query));
    const list = $("document-list");
    list.replaceChildren();
    if (!rows.length) list.append(element("p", "empty-list", "Документов с таким названием не найдено."));
    for (const d of rows) {
      const button = element("button", "document-item");
      button.type = "button";
      const body = element("span", "document-item-content");
      body.append(element("span", "document-item-title", d.title), element("span", "document-item-meta", [typeLabels[d.document_type] || "Документ", d.edition ? "ред. " + d.edition : d.contract_number].filter(Boolean).join(" · ")));
      button.append(element("span", "document-item-icon", "▤"), body);
      button.addEventListener("click", () => openDocument(d.document_id));
      list.append(button);
    }
  }
  async function openCitation(citation) { return openDocument(citation.document_id, citation); }
  async function openDocument(id, citation = null) {
    setPanel(true);
    const token = ++documentToken;
    $("document-library").hidden = true;
    $("document-view").hidden = false;
    $("panel-eyebrow").textContent = citation ? "ПОДТВЕРЖДАЮЩЕЕ ПОЛОЖЕНИЕ" : "ИСХОДНЫЙ ТЕКСТ";
    $("document-title").textContent = "Загружаем документ…";
    $("document-type").textContent = "";
    $("document-meta").textContent = "";
    $("document-text").replaceChildren();
    $("document-message").hidden = true;
    $("download-document").hidden = true;
    $("jump-to-quote").hidden = true;
    quoteMark = null;
    try {
      const [doc, chunkResult] = await Promise.all([
        request("/api/documents/" + encodeURIComponent(id)),
        citation ? request("/api/chunks/" + encodeURIComponent(citation.chunk_id)) : Promise.resolve(null),
      ]);
      if (token !== documentToken) return;
      $("document-title").textContent = doc.title;
      $("document-type").textContent = typeLabels[doc.document_type] || "Документ";
      $("document-meta").textContent = [citation?.section, citation?.edition || (doc.edition ? "Редакция от " + doc.edition : null), doc.source_file].filter(Boolean).join(" · ");
      const download = $("download-document");
      download.hidden = false;
      download.href = "/api/documents/" + encodeURIComponent(id) + "/download";
      if (citation) {
        // Контракт задаёт Unicode code points; JS slice по UTF-16 здесь неверен.
        const chars = Array.from(doc.text);
        const { char_start: start, char_end: end } = citation.coordinates;
        const chunk = chunkResult.chunk;
        const valid = chunk.document_id === id && Number.isInteger(start) && Number.isInteger(end) && start >= chunk.start_char && end <= chunk.end_char && start < end && chars.slice(start, end).join("") === citation.quote;
        if (valid) {
          quoteMark = element("mark", "", chars.slice(start, end).join(""));
          $("document-text").append(document.createTextNode(chars.slice(0, start).join("")), quoteMark, document.createTextNode(chars.slice(end).join("")));
          $("jump-to-quote").hidden = false;
          requestAnimationFrame(() => quoteMark?.scrollIntoView({ block: "center", behavior: "instant" }));
        } else {
          $("document-text").textContent = doc.text;
          $("document-message").hidden = false;
          $("document-message").textContent = "Цитата не совпала с текущим текстом. Проверьте документ вручную: база могла обновиться.";
        }
      } else {
        $("document-text").textContent = doc.text;
        $("document-scroll").scrollTop = 0;
      }
    } catch (error) {
      if (token !== documentToken) return;
      $("document-title").textContent = "Не удалось открыть документ";
      $("document-message").hidden = false;
      $("document-message").textContent = error.message;
    }
  }
  function closeSidebar() {
    $("sidebar").classList.remove("open");
    $("sidebar-backdrop").hidden = true;
    $("menu-button").setAttribute("aria-expanded", "false");
    updatePanelMode();
  }
  $("new-chat").addEventListener("click", newConversation);
  $("chat-form").addEventListener("submit", (event) => { event.preventDefault(); sendQuestion(); });
  $("question").addEventListener("input", () => {
    $("question").style.height = "auto";
    $("question").style.height = Math.min($("question").scrollHeight, 150) + "px";
    updateControls();
  });
  $("question").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing && !window.matchMedia("(pointer: coarse)").matches) {
      event.preventDefault();
      sendQuestion();
    }
  });
  $("as-of").addEventListener("change", () => { current().asOf = $("as-of").value; persist(); });
  for (const suggestion of document.querySelectorAll(".suggestion")) suggestion.addEventListener("click", () => { $("question").value = suggestion.dataset.question; $("question").focus(); updateControls(); });
  $("sources-toggle").addEventListener("click", () => $("document-panel").hidden ? openLibrary() : setPanel(false));
  $("library-button").addEventListener("click", openLibrary);
  $("close-panel").addEventListener("click", () => { documentToken++; setPanel(false); });
  $("back-to-library").addEventListener("click", openLibrary);
  $("document-query").addEventListener("input", renderDocuments);
  $("jump-to-quote").addEventListener("click", () => quoteMark?.scrollIntoView({ block: "center", behavior: "smooth" }));
  $("menu-button").addEventListener("click", () => {
    $("sidebar").classList.add("open");
    $("sidebar-backdrop").hidden = false;
    $("menu-button").setAttribute("aria-expanded", "true");
    updatePanelMode();
    $("new-chat").focus();
  });
  $("sidebar-backdrop").addEventListener("click", closeSidebar);
  window.addEventListener("resize", () => { if (window.innerWidth > 760) closeSidebar(); updatePanelMode(); });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      if (!$("document-panel").hidden) { documentToken++; setPanel(false); }
      else if ($("sidebar").classList.contains("open")) { closeSidebar(); $("menu-button").focus(); }
    }
    const modal = $("document-panel").getAttribute("aria-modal") ? $("document-panel") : ($("sidebar").classList.contains("open") ? $("sidebar") : null);
    if (event.key === "Tab" && modal) {
      const focusable = [...modal.querySelectorAll("a[href], button:not([disabled]), input:not([disabled])")].filter((e) => e.getClientRects().length);
      const first = focusable[0], last = focusable.at(-1);
      if (event.shiftKey && document.activeElement === first) { last?.focus(); event.preventDefault(); }
      else if (!event.shiftKey && document.activeElement === last) { first?.focus(); event.preventDefault(); }
    }
  });
  restore();
  render();
  refreshStatus();
})();
