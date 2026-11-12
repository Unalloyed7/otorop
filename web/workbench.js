"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const labels = {
    confirmed: "Подтверждено",
    conditional: "Подтверждено при условиях",
    contradiction: "Найдено противоречие",
    insufficient: "Недостаточно оснований"
  };
  const roles = {
    support: "Основание",
    restriction: "Ограничение",
    contradiction: "Противоречие",
    amendment: "Изменение",
    exception: "Исключение",
    other_edition: "Другая редакция",
    procedure: "Порядок действий"
  };
  const node = (tag, text, cls) => {
    const n = document.createElement(tag);
    if (text !== undefined) n.textContent = text;
    if (cls) n.className = cls;
    return n;
  };
  const button = (text, action) => {
    const b = node("button", text, "text-button");
    b.type = "button";
    b.addEventListener("click", action);
    return b;
  };
  const api = async (path, body, method = "POST") => {
    const r = await fetch(path, {
      method: body === undefined ? "GET" : method,
      headers: body === undefined ? {} : {
        "Content-Type": "application/json"
      },
      body: body === undefined ? undefined : JSON.stringify(body),
      cache: "no-store"
    });
    if (!r.ok) {
      let detail;
      try {
        detail = (await r.json()).detail;
      } catch {}
      throw new Error(typeof detail === "string" ? detail : "Не удалось выполнить действие.");
    }
    return r.json();
  };

  function citation(c, open) {
    const b = node("button", undefined, "proof-link");
    b.type = "button";
    b.append(node("strong", (roles[c.role] || "Источник") + " · " + c.document_title), node("span", c
      .section + " · " + c.edition), node("q", c.quote));
    b.addEventListener("click", () => open(c));
    return b;
  }

  function statementCard(statement, context) {
    const card = node("details", undefined, "statement-card " + statement.check.status);
    const summary = node("summary");
    summary.append(node("span", labels[statement.check.status], "check-badge"), node("strong", statement
      .text));
    card.append(summary);
    const body = node("div", undefined, "statement-content");
    if (!statement.published) body.append(node("p", "Не включено в проверенный ответ.",
    "unpublished-note"));
    const details = node("dl", undefined, "rule-details");
    const value = statement.value || {},
      subject = statement.subject || {};
    for (const [title, text] of [
        ["Субъект", [subject.contract_number, subject.counterparty, subject.employee].filter(Boolean).join(
          " · ")],
        ["Правило", {
          duty: "Обязанность",
          permission: "Разрешение",
          restriction: "Ограничение",
          information: "Положение"
        } [statement.rule]],
        ["Значение", value.description],
        ["Срок / сумма", [value.duration, value.unit, value.day_type === "calendar" ? "календарные" : value
          .day_type === "working" ? "рабочие" : "", value.amount, value.currency
        ].filter(Boolean).join(" ")],
        ["Событие отсчёта", value.counted_from],
        ["Период действия", statement.period ? [statement.period.valid_from || "начало не установлено",
          statement.period.valid_to ? "до " + statement.period.valid_to + " (не включая)" :
          "окончание не установлено"
        ].join(" — ") : null]
      ]) {
      details.append(node("dt", title), node("dd", text || "Не установлено"));
    }
    body.append(details);
    if (statement.conditions?.length) {
      body.append(node("h4", "Условия применения"));
      for (const c of statement.conditions) {
        body.append(node("p", ({
          met: "Установлено",
          not_met: "Не выполнено",
          unknown: "Нужно установить"
        } [c.status]) + ": " + c.text));
        for (const ref of c.citations || []) body.append(citation(ref, context.openCitation));
      }
    }
    body.append(node("h4", "Подтверждающие положения"));
    for (const c of statement.supports || []) body.append(citation(c, context.openCitation));
    if (!statement.supports?.length) body.append(node("p", "Подтверждающие ссылки не прошли проверку."));
    if (statement.limitations?.length) {
      body.append(node("h4", "Ограничения и связанные изменения"));
      for (const c of statement.limitations) body.append(citation(c, context.openCitation));
    }
    for (const reason of statement.check.reasons || []) body.append(node("p", reason, "check-reason"));
    body.append(node("p", statement.check.semantic_checked ?
      "Смысловая связь проверена моделью; проверка может ошибаться." :
      "Смысловая проверка не выполнена.", "check-reason"));
    const refs = [...(statement.supports || []), ...(statement.limitations || [])];
    if (refs.length > 1) body.append(button("Сравнить положения рядом", () => showComparison(statement,
      context)));
    body.append(button("Изменить обстоятельства и проверить", () => {
      $("scenario-settings").open = true;
      $("scenario-event").focus();
    }));
    card.append(body);
    return card;
  }

  function dateText(value) {
    if (typeof value === "string") return value;
    return value?.label || (value?.precision === "year" ? value.start.slice(0, 4) : value?.precision ===
      "month" ? value.start.slice(0, 7) : value?.start) || "";
  }

  function readScenario() {
    const result = {};
    for (const kind of ["contract_signed", "specification_signed", "delivery", "event"]) {
      const v = $("scenario-" + kind).value.trim();
      if (v) result[kind] = v;
    }
    const facts = {};
    for (const key of ["standard_form_unchanged", "foreign_counterparty"]) {
      const v = $("fact-" + key).value;
      if (v) facts[key] = v === "true";
    }
    result.facts = facts;
    return result;
  }

  function writeScenario(value = {}) {
    for (const kind of ["contract_signed", "specification_signed", "delivery", "event"]) $("scenario-" +
      kind).value = dateText(value[kind]);
    for (const key of ["standard_form_unchanged", "foreign_counterparty"]) $("fact-" + key).value = value
      .facts?.[key] === true ? "true" : value.facts?.[key] === false ? "false" : "";
  }

  function dialog(title) {
    const d = node("dialog", undefined, "workbench-dialog");
    const header = node("header");
    header.append(node("h2", title), button("Закрыть", () => d.close()));
    d.append(header);
    d.addEventListener("close", () => d.remove());
    document.body.append(d);
    d.showModal();
    return d;
  }

  function showComparison(statement, context) {
    const d = dialog("Сравнение оснований и условий");
    const refs = [...(statement.supports || []), ...(statement.limitations || [])];
    d.append(node("p", "Применимость определяет движок. Здесь показаны положения выбранного ответа."));
    const grid = node("div", undefined, "comparison-grid");
    for (let column = 0; column < 2; column++) {
      const pane = node("section");
      const select = node("select");
      select.setAttribute("aria-label", column ? "Изменение или ограничение" : "Исходное положение");
      refs.forEach((r, i) => {
        const option = node("option", (roles[r.role] || "Источник") + ": " + r.document_title + ", " + r
          .section);
        option.value = i;
        select.append(option);
      });
      select.value = String(column ? Math.max(1, refs.findIndex(r => r.role === "amendment")) : 0);
      const content = node("div");
      const render = () => {
        const r = refs[Number(select.value)];
        content.replaceChildren(node("h3", r.document_title), node("p", r.section + " · " + r.edition),
          node("pre", r.quote));
        for (const condition of r.applicability?.conditions || []) content.append(node("p", "Условие: " +
          condition));
        content.append(button("Открыть точную цитату в оригинале", () => {
          d.close();
          context.openCitation(r);
        }));
      };
      select.addEventListener("change", render);
      pane.append(select, content);
      grid.append(pane);
      render();
    }
    d.append(grid, node("h3", "Обстоятельства сценария"));
    const scenario = readScenario();
    for (const kind of ["contract_signed", "specification_signed", "delivery", "event"]) {
      if (scenario[kind]) d.append(node("p", ({
        contract_signed: "Подписание договора",
        specification_signed: "Подписание спецификации",
        delivery: "Поставка",
        event: "Событие"
      } [kind]) + ": " + scenario[kind]));
    }
    for (const c of statement.conditions || []) d.append(node("p", c.text + ": " + ({
      met: "установлено",
      not_met: "не выполнено",
      unknown: "неизвестно"
    } [c.status])));
    d.append(button("Изменить сценарий и пересчитать", () => {
      d.close();
      $("scenario-settings").open = true;
      $("scenario-event").focus();
    }), button("Проверить текущий сценарий заново", () => {
      d.close();
      context.recheck();
    }));
  }

  function sourceKeys(record) {
    return new Map((record.sources || []).map(s => [
      [s.document_title, s.section, s.edition, s.text].join("\0"), s
    ]));
  }

  function difference(left, right) {
    return [...left.keys()].filter(k => !right.has(k));
  }
  async function openDiagnostics() {
    const d = dialog("Анализ локальных запусков");
    const status = node("p",
      "Выберите два отчёта или импортируйте JSONL. Официальные баллы учитываются отдельно.");
    d.append(status);
    const input = node("input");
    input.type = "file";
    input.accept = ".jsonl,.json";
    input.multiple = true;
    input.setAttribute("aria-label", "Импорт локальных диагностических отчётов");
    const selectors = node("div", undefined, "comparison-grid"),
      left = node("select"),
      right = node("select");
    left.setAttribute("aria-label", "Первый запуск");
    right.setAttribute("aria-label", "Второй запуск");
    selectors.append(left, right);
    const content = node("div");
    async function refresh() {
      const data = await api("/api/runs");
      for (const select of [left, right]) {
        const old = select.value;
        select.replaceChildren();
        select.append(node("option", "— Выберите запуск —"));
        select.firstChild.value = "";
        for (const run of data.runs) {
          const option = node("option", run.name);
          option.value = run.id;
          select.append(option);
        }
        if ([...select.options].some(o => o.value === old)) select.value = old;
      }
    }
    input.addEventListener("change", async () => {
      try {
        for (const file of input.files) {
          if (file.size > 7_000_000) throw new Error("Отчёт слишком большой: максимум 7 МБ.");
          const text = await file.text();
          let records, configuration = {};
          if (file.name.endsWith(".jsonl")) {
            records = text.split("\n").filter(line => line.trim()).map(line => JSON.parse(line));
          } else {
            const parsed = JSON.parse(text);
            records = Array.isArray(parsed) ? parsed : parsed.records;
            configuration = parsed.configuration || {};
          }
          await api("/api/runs/import", {
            name: file.name,
            records,
            configuration
          });
        }
        await refresh();
        status.textContent = "Отчёты импортированы. Выберите запуски для сравнения.";
      } catch (e) {
        status.textContent = e.message;
      }
    });
    async function compare() {
      try {
        if (!left.value) throw new Error("Выберите хотя бы первый запуск.");
        const [a, b] = await Promise.all([api("/api/runs/" + left.value), right.value ? api(
          "/api/runs/" + right.value) : Promise.resolve(null)]);
        content.replaceChildren();
        const configs = node("div", undefined, "comparison-grid");
        for (const run of [a, b].filter(Boolean)) {
          const p = node("section");
          p.append(node("h3", run.name), node("pre", Object.keys(run.configuration).length ? JSON
            .stringify(run.configuration, null, 2) : "Версия и настройки в отчёте не записаны"));
          configs.append(p);
        }
        content.append(configs);
        const other = new Map((b?.records || []).map(r => [r.id, r]));
        for (const record of a.records) {
          const paired = other.get(record.id);
          const review = paired || record,
            run = paired ? b : a;
          const row = node("details", undefined, "run-case");
          row.append(node("summary", record.id + " · " + record.answer.status + (paired ? " → " + paired
            .answer.status : "")), node("p", record.question));
          for (const [label, r] of [
              [a.name, record],
              [b?.name, paired]
            ].filter(x => x[1])) {
            row.append(node("h4", label), node("p", r.answer.text), node("p",
              `Задержка: ${r.duration_ms??"нет данных"} мс · вызовы: ${JSON.stringify(r.calls)} · ошибки формата: ${r.format_errors}`
              ));
            for (const issue of r.rejections || []) row.append(node("p", "Отброшено: " + issue.reason));
            for (const error of r.errors || []) row.append(node("p", "Ошибка: " + error.kind + " · " + (
              error.message || "")));
            for (const format of r.format_details || []) {
              const detail = node("details");
              detail.append(node("summary", "Ошибка формата " + (format.schema || "модели")), node(
                "pre", JSON.stringify(format.errors, null, 2)), node("pre", format.output));
              row.append(detail);
            }
          }
          if (paired) {
            const sa = sourceKeys(record),
              sb = sourceKeys(paired);
            row.append(node("p",
              `Источники: добавлено ${difference(sb,sa).length}, убрано ${difference(sa,sb).length}`
              ));
            for (const k of difference(sb, sa)) row.append(node("p", "+ " + sb.get(k).document_title +
              ", " + sb.get(k).section));
            for (const k of difference(sa, sb)) row.append(node("p", "− " + sa.get(k).document_title +
              ", " + sa.get(k).section));
            const oldClaims = new Set((record.answer.statements || []).map(s => s.text)),
              newClaims = new Set((paired.answer.statements || []).map(s => s.text));
            for (const t of newClaims)
              if (!oldClaims.has(t)) row.append(node("p", "Добавлено утверждение: " + t));
            for (const t of oldClaims)
              if (!newClaims.has(t)) row.append(node("p", "Убрано утверждение: " + t));
          }
          const rating = node("select");
          rating.setAttribute("aria-label", "Ручная оценка корректности " + record.id);
          for (const [v, t] of Object.entries({
              unknown: "Не оценено",
              correct: "Корректно",
              partial: "Частично корректно",
              wrong: "Некорректно"
            })) {
            const option = node("option", t);
            option.value = v;
            rating.append(option);
          }
          rating.value = review.rating.correctness;
          const refusal = node("select");
          refusal.setAttribute("aria-label", "Лишний отказ " + record.id);
          for (const [v, t] of [
              ["", "Отказ не оценён"],
              ["true", "Есть лишний отказ"],
              ["false", "Лишнего отказа нет"]
            ]) {
            const o = node("option", t);
            o.value = v;
            refusal.append(o);
          }
          refusal.value = review.rating.unnecessary_refusal === null ? "" : String(review.rating
            .unnecessary_refusal);
          const note = node("textarea");
          note.value = review.rating.note;
          note.maxLength = 2000;
          note.setAttribute("aria-label", "Комментарий к оценке " + record.id);
          row.append(node("p", "Ручная оценка: " + run.name), rating, refusal, note, button(
            "Сохранить оценку", async () => {
              try {
                await api(
                `/api/runs/${run.run_id}/cases/${encodeURIComponent(record.id)}/rating`, {
                  correctness: rating.value,
                  unnecessary_refusal: refusal.value === "" ? null : refusal.value === "true",
                  note: note.value
                }, "PATCH");
                status.textContent = "Оценка сохранена.";
              } catch (e) {
                status.textContent = e.message;
              }
            }));
          content.append(row);
        }
        if (b)
          for (const r of b.records)
            if (!a.records.some(old => old.id === r.id)) content.append(node("p",
              "Только во втором запуске: " + r.id));
      } catch (e) {
        status.textContent = e.message;
      }
    }
    d.append(input, selectors, button("Показать различия", compare), content);
    const scores = node("details", undefined, "official-scores");
    scores.append(node("summary", "Официальные агрегированные баллы"), node("p",
      "Это запись результатов платформы. Скрытые ответы и серверные логи сюда не импортируются."));
    const archive = node("input"),
      score = node("input"),
      duration = node("input");
    archive.placeholder = "Имя архива";
    score.placeholder = "Балл, как показан платформой";
    duration.placeholder = "Показанная длительность, сек";
    score.type = duration.type = "number";
    score.step = duration.step = "any";
    const note = node("input");
    note.placeholder = "Шкала / заметка, например Public";
    note.maxLength = 2000;
    const listing = node("div");
    async function loadScores() {
      const data = await api("/api/official-scores");
      listing.replaceChildren(...data.scores.map(s => node("p", s.archive + ": " + s.score + " · " + (s
        .duration_seconds ?? "—") + " сек" + (s.note ? " · " + s.note : ""))));
    }
    scores.append(archive, score, duration, note, button("Записать официальный результат", async () => {
      try {
        if (!archive.value.trim() || score.value === "") throw new Error("Укажите архив и балл.");
        await api("/api/official-scores", {
          archive: archive.value,
          score: Number(score.value),
          duration_seconds: duration.value ? Number(duration.value) : null,
          note: note.value
        });
        await loadScores();
      } catch (e) {
        status.textContent = e.message;
      }
    }), listing);
    d.append(scores);
    try {
      await refresh();
      await loadScores();
    } catch (e) {
      status.textContent = e.message;
    }
  }
  window.LegalWorkbench = {
    statementCard,
    readScenario,
    writeScenario,
    openDiagnostics,
    showComparison
  };
})();
