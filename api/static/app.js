"use strict";

const $ = (id) => document.getElementById(id);
const form = $("application-form");
const submit = $("submit-button");
const numericFormat = new Intl.NumberFormat("fr-FR", {maximumFractionDigits: 1});
let modelReady = false;
let pending = false;

function showError(message) {
  $("form-error").textContent = message;
  $("form-error").hidden = !message;
}

function clearResult() {
  $("analysis-result").hidden = true;
  $("result-version").hidden = true;
  $("empty-state").hidden = false;
  showError("");
  $("result-announcement").textContent = "";
}

function setPending(value) {
  pending = value;
  form.querySelectorAll("input, button").forEach((control) => { control.disabled = value; });
  $("fill-example").disabled = value;
  submit.disabled = value || !modelReady;
  $("loading-state").hidden = !value;
  document.querySelector(".result-panel").setAttribute("aria-busy", String(value));
  submit.firstChild.textContent = value ? "Analyse en cours… " : "Analyser le dossier ";
}

async function jsonRequest(path, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 90000);
  try {
    const response = await fetch(path, {cache: "no-store", ...options, signal: controller.signal});
    if (!response.ok) {
      if (response.status === 503) throw new Error("Le modèle est indisponible. Actualisez son état, puis réessayez.");
      if (response.status === 422) throw new Error("Certaines données ne respectent pas le format attendu. Vérifiez les montants, les scores et la référence du dossier.");
      throw new Error("Le service n’a pas pu terminer l’analyse. Réessayez dans un instant.");
    }
    return await response.json();
  } catch (error) {
    if (error.name === "AbortError") throw new Error("Le délai d’analyse est dépassé. Réessayez dans un instant.");
    if (error instanceof TypeError) throw new Error("Le service est injoignable. Vérifiez la connexion, puis réessayez.");
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

async function refreshModel() {
  $("refresh-model").disabled = true;
  try {
    const info = await jsonRequest("/model-info");
    modelReady = true;
    $("model-status").className = "status ready";
    $("model-status").textContent = "Disponible";
    $("model-version").textContent = `v${info.version}`;
    $("model-auc").textContent = Number.isFinite(info.metrics.validation_auc) ? info.metrics.validation_auc.toFixed(4).replace(".", ",") : "—";
    $("model-threshold").textContent = numericFormat.format(info.threshold * 100) + (info.calibrated === true ? " %" : " / 100");
    $("model-note").textContent = info.calibrated === true ? "Version chargée par le service · probabilité calibrée" : "Version chargée par le service · score non calibré";
  } catch (error) {
    modelReady = false;
    $("model-status").className = "status unavailable";
    $("model-status").textContent = "Indisponible";
    $("model-version").textContent = "—";
    $("model-auc").textContent = "—";
    $("model-threshold").textContent = "—";
    $("model-note").textContent = error.message;
  } finally {
    submit.disabled = pending || !modelReady;
    $("refresh-model").disabled = false;
  }
}

function buildPayload() {
  const payload = {customer_id: $("customer_id").value.trim()};
  if (!payload.customer_id) throw new Error("Renseignez une référence de dossier.");
  const history = {};
  form.querySelectorAll('input[type="number"]').forEach((input) => {
    if (input.value === "") return;
    const value = input.valueAsNumber;
    if (!Number.isFinite(value)) throw new Error("Renseignez uniquement des nombres valides.");
    if (input.name === "employment_years") {
      const days = Math.round(value * -365.25);
      if (!Number.isSafeInteger(days)) throw new Error("L’ancienneté professionnelle est trop élevée.");
      payload.DAYS_EMPLOYED = days;
    } else if (input.hasAttribute("data-history")) {
      history[input.name] = value;
    } else {
      payload[input.name] = value;
    }
  });
  if (Object.keys(history).length) payload.history = history;
  return payload;
}

function renderResult(result) {
  if (!Number.isFinite(result.default_score) || result.default_score < 0 || result.default_score > 1 || !Number.isFinite(result.threshold) || result.threshold <= 0 || result.threshold >= 1 || ![0, 1].includes(result.predicted_default) || !Array.isArray(result.reasons) || result.reasons.length !== 3 || !Array.isArray(result.missing_inputs)) {
    throw new Error("Le service a renvoyé une analyse incomplète. Réessayez.");
  }
  if (result.reasons.some((reason) => !Number.isFinite(reason.contribution_raw_score) || typeof reason.label !== "string" || typeof reason.direction !== "string")) {
    throw new Error("Les explications reçues sont incomplètes. Réessayez.");
  }
  const refusal = result.predicted_default === 1;
  const calibrated = result.calibrated === true;
  $("score-unit").textContent = calibrated ? "%" : "/ 100";
  $("score-caption").textContent = calibrated ? "Probabilité estimée de défaut · calibrée sur Home Credit" : "Score du modèle · non calibré";
  $("score-note").textContent = calibrated ? "Calibration vérifiée sur des dossiers Home Credit réservés. Les dossiers incomplets sont imputés ; cette probabilité nécessite une validation sur la population d’utilisation." : "Ce score n’est pas une probabilité de défaut calibrée. La décision affichée est celle du modèle au seuil configuré.";
  $("decision").textContent = refusal ? "Refus selon le seuil du modèle" : "Acceptation selon le seuil du modèle";
  $("decision").className = refusal ? "decision refusal" : "decision";
  $("score-value").textContent = numericFormat.format(result.default_score * 100);
  $("score-fill").style.width = `${result.default_score * 100}%`;
  $("score-fill").className = refusal ? "refusal" : "";
  $("threshold-marker").style.left = `${result.threshold * 100}%`;
  $("threshold-label").textContent = `Seuil : ${numericFormat.format(result.threshold * 100)}${calibrated ? " %" : ""}`;
  $("score-track").setAttribute("aria-label", `${calibrated ? "Probabilité estimée de défaut" : "Score"} ${numericFormat.format(result.default_score * 100)} ${calibrated ? "pour cent" : "sur 100"} ; seuil ${numericFormat.format(result.threshold * 100)}`);
  $("result-version").textContent = `Modèle v${result.model_version}`;
  $("result-version").hidden = false;
  const max = Math.max(...result.reasons.map((reason) => Math.abs(reason.contribution_raw_score)), Number.EPSILON);
  $("reasons").replaceChildren();
  result.reasons.forEach((reason) => {
    const item = document.createElement("li");
    const label = document.createElement("div");
    label.className = "reason-label";
    label.textContent = reason.label;
    const direction = document.createElement("div");
    direction.className = reason.contribution_raw_score > 0 ? "reason-direction increase" : "reason-direction";
    direction.textContent = `${reason.contribution_raw_score > 0 ? "↑" : reason.contribution_raw_score < 0 ? "↓" : "−"} ${reason.direction}`;
    const track = document.createElement("div");
    track.className = "contribution-track";
    track.setAttribute("aria-hidden", "true");
    const bar = document.createElement("div");
    bar.style.width = `${Math.abs(reason.contribution_raw_score) / max * 100}%`;
    bar.className = reason.contribution_raw_score > 0 ? "increase" : "";
    track.append(bar);
    item.append(label, direction, track);
    $("reasons").append(item);
  });
  $("missing-title").textContent = `${result.missing_inputs.length} champs absents · valeurs imputées`;
  $("missing-fields").replaceChildren();
  result.missing_inputs.forEach((name) => {
    const item = document.createElement("li");
    item.textContent = name.replaceAll("_", " ").toLowerCase();
    $("missing-fields").append(item);
  });
  $("request-id").textContent = result.request_id;
  $("empty-state").hidden = true;
  $("analysis-result").hidden = false;
  $("result-announcement").textContent = `Analyse terminée. ${$("decision").textContent}. ${calibrated ? "Probabilité estimée de défaut" : "Score"} : ${numericFormat.format(result.default_score * 100)} ${calibrated ? "pour cent" : "sur 100"}.`;
  if (window.innerWidth <= 680) {
    document.querySelector(".result-panel").scrollIntoView({behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start"});
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (pending || !modelReady || !form.reportValidity()) return;
  let payload;
  try { payload = buildPayload(); } catch (error) { showError(error.message); return; }
  clearResult();
  $("empty-state").hidden = true;
  setPending(true);
  try {
    const result = await jsonRequest("/explain", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
    renderResult(result);
  } catch (error) {
    $("empty-state").hidden = false;
    showError(error.message);
  } finally {
    setPending(false);
  }
});

$("fill-example").addEventListener("click", () => {
  form.reset();
  const values = {customer_id: "EXEMPLE-FICTIF", AMT_INCOME_TOTAL: 180000, AMT_CREDIT: 500000, AMT_ANNUITY: 30000, AMT_GOODS_PRICE: 450000, employment_years: 5, EXT_SOURCE_1: 0.5, EXT_SOURCE_2: 0.6, EXT_SOURCE_3: 0.7};
  Object.entries(values).forEach(([name, value]) => { $(name).value = value; });
  clearResult();
});
form.addEventListener("input", clearResult);
form.addEventListener("reset", clearResult);
$("refresh-model").addEventListener("click", refreshModel);
refreshModel();
