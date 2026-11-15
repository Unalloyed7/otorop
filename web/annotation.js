"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const STORAGE_KEY = "otorop-annotation-review-v1";
  const dispositionLabels = {
    answer: "Ответ с основаниями",
    partial: "Частичный ответ",
    conditional: "Ответ с условием",
    correct_false_premise: "Исправить ложную предпосылку",
    insufficient: "Недостаточно оснований"
  };
  let batch;
  let reviews = {};
  let activeIndex = 0;
  let saveTimer;

  const element = (tag, text, className) => {
    const item = document.createElement(tag);
    if (text !== undefined) item.textContent = text;
    if (className) item.className = className;
    return item;
  };

  function readReviews() {
    try {
      const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
      if (saved && typeof saved === "object" && !Array.isArray(saved)) reviews = saved;
    } catch {
      reviews = {};
      $("save-status").textContent = "Не удалось прочитать сохранённые отметки. Новые решения будут сохранены заново.";
      $("save-status").classList.add("warning");
    }
  }

  function saveReview(caseId, patch) {
    const current = reviews[caseId] || {};
    reviews[caseId] = { ...current, ...patch, updated_at: new Date().toISOString() };
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(reviews));
      $("save-status").textContent = "Отметка сохранена в этом браузере.";
      $("save-status").classList.remove("warning");
    } catch {
      $("save-status").textContent = "Браузер не смог сохранить отметку. Скачай решения перед закрытием страницы.";
      $("save-status").classList.add("warning");
    }
    renderNavigation();
    renderProgress();
  }

  function renderProgress() {
    const counts = { confirmed: 0, needs_revision: 0, unclear: 0, pending: 0 };
    for (const item of batch.cases) {
      const decision = reviews[item.id]?.decision;
      counts[Object.hasOwn(counts, decision) ? decision : "pending"]++;
    }
    $("progress-summary").textContent =
      `Подтверждено ${counts.confirmed} из ${batch.cases.length} · правка ${counts.needs_revision} · неясно ${counts.unclear} · без решения ${counts.pending}`;
  }

  function renderNavigation() {
    const list = $("case-list");
    list.replaceChildren();
    const query = $("case-filter").value.trim().toLocaleLowerCase("ru-RU");
    const visible = batch.cases.map((item, index) => ({ item, index })).filter(({ item }) =>
      !query || [item.id, item.title, item.category, item.question].some((value) =>
        value.toLocaleLowerCase("ru-RU").includes(query)));
    if (!visible.length) {
      list.append(element("p", "Подходящих кейсов нет.", "empty-case-list"));
      return;
    }
    visible.forEach(({ item, index }) => {
      const decision = reviews[item.id]?.decision;
      const label = decision === "confirmed" ? "Подтверждён" :
        decision === "needs_revision" ? "Нужна правка" :
        decision === "unclear" ? "Неясно" : "Не проверен";
      const button = element("button", undefined, "case-nav-item");
      button.type = "button";
      button.setAttribute("aria-current", index === activeIndex ? "true" : "false");
      button.append(element("strong", item.id), element("span", item.title), element("small", label));
      button.addEventListener("click", () => {
        activeIndex = index;
        renderCase();
      });
      list.append(button);
    });
  }

  function appendList(target, values) {
    target.replaceChildren();
    for (const value of values || []) target.append(element("li", value));
    target.hidden = !values?.length;
  }

  function renderCase() {
    const item = batch.cases[activeIndex];
    if (!item) return;
    const review = reviews[item.id] || {};
    $("case-kicker").textContent = `${item.id} · ${item.category}`;
    $("case-title").textContent = item.title;
    $("corpus-version").textContent = `Пакет: ${batch.batch_id} · SHA-256 исходной базы: ${batch.source_sha256}`;
    $("case-question").textContent = item.question;
    $("case-disposition").textContent = dispositionLabels[item.expected_disposition] || item.expected_disposition;
    appendList($("required-claims"), item.required_claims);
    appendList($("must-not-claim"), item.must_not_claim);
    $("limitation-block").hidden = !item.required_limitation;
    $("required-limitation").textContent = item.required_limitation || "";
    $("source-list").replaceChildren();
    for (const source of item.sources || []) {
      const card = element("blockquote", undefined, "source-card");
      card.append(element("p", `${source.document} · ${source.section}`), element("q", source.quote));
      $("source-list").append(card);
    }
    $("review-note").value = review.note || "";
    $("case-content").hidden = false;
    $("previous-case").disabled = activeIndex === 0;
    $("next-case").disabled = activeIndex === batch.cases.length - 1;
    for (const button of document.querySelectorAll("[data-decision]")) {
      button.setAttribute("aria-pressed", String(button.dataset.decision === review.decision));
    }
    $("save-status").textContent = review.updated_at ?
      `Последняя отметка сохранена ${new Date(review.updated_at).toLocaleString("ru-RU")}.` :
      "Решения сохраняются только в этом браузере.";
    renderNavigation();
  }

  function exportReview() {
    const output = {
      schema_version: 1,
      batch_id: batch.batch_id,
      source_sha256: batch.source_sha256,
      exported_at: new Date().toISOString(),
      cases: batch.cases.map((item) => ({
        proposal: item,
        review: reviews[item.id] || { decision: "pending", note: "" }
      }))
    };
    const url = URL.createObjectURL(new Blob([JSON.stringify(output, null, 2)], {
      type: "application/json"
    }));
    const link = element("a");
    link.href = url;
    link.download = `${batch.batch_id}-review.json`;
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  async function start() {
    try {
      const response = await fetch("/api/annotation-batch", { cache: "no-store" });
      if (!response.ok) throw new Error("Не удалось загрузить пакет разметки.");
      batch = await response.json();
      if (!Array.isArray(batch.cases) || !batch.cases.length) {
        throw new Error("В пакете нет сценариев для проверки.");
      }
      readReviews();
      $("loading").hidden = true;
      $("export-review").disabled = false;
      renderProgress();
      renderCase();
    } catch (error) {
      $("loading").hidden = true;
      $("load-error").textContent = `${error.message} Проверь, что запущено приложение из корня репозитория.`;
      $("load-error").hidden = false;
    }
  }

  document.querySelectorAll("[data-decision]").forEach((button) => {
    button.addEventListener("click", () => {
      const caseId = batch.cases[activeIndex].id;
      saveReview(caseId, { decision: button.dataset.decision, note: $("review-note").value.trim() });
      for (const control of document.querySelectorAll("[data-decision]")) {
        control.setAttribute("aria-pressed", String(control === button));
      }
    });
  });
  $("review-note").addEventListener("input", () => {
    clearTimeout(saveTimer);
    saveTimer = setTimeout(() => {
      const item = batch?.cases[activeIndex];
      if (item && reviews[item.id]?.decision) saveReview(item.id, { note: $("review-note").value.trim() });
    }, 350);
  });
  $("previous-case").addEventListener("click", () => {
    if (activeIndex > 0) {
      activeIndex--;
      renderCase();
    }
  });
  $("next-case").addEventListener("click", () => {
    if (activeIndex < batch.cases.length - 1) {
      activeIndex++;
      renderCase();
    }
  });
  $("export-review").addEventListener("click", exportReview);
  $("case-filter").addEventListener("input", renderNavigation);
  start();
})();
