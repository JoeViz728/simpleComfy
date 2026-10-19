const $ = (id) => document.getElementById(id);

const MODES = [
  { id: "txt2img", label: "Text → Image" },
  { id: "txt2vid", label: "Text → Video" },
  { id: "img2vid", label: "Image → Video" },
  { id: "vid2vid", label: "Video → Video" },
];

/** Ops under Video → Video (map to backend smooth / continue / re-gen final). */
const VID2VID_OPS = [
  {
    id: "smooth",
    label: "Smooth",
    description: "Frame-interpolate (cheap) — 2× or 4× in-betweens",
  },
  {
    id: "continue",
    label: "Continue",
    description: "Extend from last frame + concat",
  },
  {
    id: "finalize",
    label: "Final",
    description: "Re-run the recipe at full Final quality",
  },
];

let presets = [];
let allEffects = [];
let capabilities = { modalities: [], video_packs: [] };
let modality = "txt2img";
/** @type {"smooth"|"continue"|"finalize"} */
let vid2vidOp = "smooth";
/** @type {2|4} */
let smoothFactor = 2;
/** @type {"draft"|"final"|"fast"|"quality"} */
let videoProfile = "draft";
/** @type {"auto"|"on"|"off"} context windows for long single-shot gen */
let contextMode = "auto";
/** @type {"flf_bridge"|"prefix"} */
let continueMethod = "flf_bridge";
let selected = null;
/** @type {Set<string>} */
let selectedEffects = new Set();
let galleryItems = [];
let galleryOffset = 0;
let galleryTotal = 0;
let galleryLoading = false;
let galleryHasMore = true;
/** How many cards are currently mounted in the DOM (for incremental append). */
let galleryRenderedCount = 0;
const GALLERY_PAGE = 36;
/** Optional observer that only loads video poster frames when a card is near the viewport. */
let galleryThumbObserver = null;
/** @type {object|null} */
let selectedGallery = null;
/**
 * Source still for img2vid:
 * { type:'gallery', jobId, filename, url, width?, height?, videoWidth?, videoHeight? }
 * | { type:'upload', input: name, url, width?, height?, videoWidth?, videoHeight? }
 */
let sourceImage = null;
/**
 * Source clip for vid2vid:
 * { jobId, filename, url, width?, height?, profile?, srcModality?, meta?, prompt?, negative?, seed? }
 */
let sourceVideo = null;
let galleryObserver = null;
/** Job ids submitted from this browser session (for Latest result updates). */
const sessionJobIds = new Set();
/** @type {Set<string>} */
const seenCompletedIds = new Set();
let queuePollTimer = null;
let lastQueueSnap = null;
/** @type {"create"|"history"} */
let activePage = "create";

function isVid2Vid() {
  return modality === "vid2vid";
}

function itemHasImage(item) {
  if (!item) return false;
  if (item.has_image) return true;
  return (item.outputs || []).some((o) => o && o.kind === "image" && !isVideoOutput(o));
}

function itemHasVideo(item) {
  if (!item) return false;
  if (item.has_video) return true;
  return (item.outputs || []).some((o) => isVideoOutput(o));
}

function firstVideoOf(item) {
  return (item.outputs || []).find((o) => isVideoOutput(o)) || null;
}

async function api(path, opts = {}) {
  const headers = { ...(opts.headers || {}) };
  if (opts.body && !(opts.body instanceof FormData)) {
    headers["Content-Type"] = headers["Content-Type"] || "application/json";
  }
  const res = await fetch(path, { ...opts, headers });
  const text = await res.text();
  let data;
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    data = { raw: text };
  }
  if (!res.ok) {
    const msg = data.detail
      ? typeof data.detail === "string"
        ? data.detail
        : JSON.stringify(data.detail)
      : res.statusText;
    throw new Error(msg);
  }
  return data;
}

function setStatus(msg, isErr = false) {
  const el = $("status");
  el.textContent = msg || "";
  el.classList.toggle("err", !!isErr);
}

const PHASE_LABELS = {
  queued: "Queued",
  ensuring_models: "Checking models",
  downloading: "Downloading models",
  waiting_comfy: "Waiting for ComfyUI",
  sampling: "Generating",
  saving: "Saving outputs",
  postprocess: "Post-processing",
  extract_motion_prefix: "Motion prefix",
  generate_chunk: "Continue chunk",
  flf_bridge: "FLF bridge",
  joining: "Joining clips",
  cancelling: "Stopping",
  done: "Done",
  failed: "Failed",
  cancelled: "Cancelled",
};

function setPage(page) {
  activePage = page === "history" ? "history" : "create";
  const createPage = $("page-create");
  const historyPage = $("page-history");
  const tabCreate = $("tab-create");
  const tabHistory = $("tab-history");
  if (createPage) createPage.hidden = activePage !== "create";
  if (historyPage) historyPage.hidden = activePage !== "history";
  if (tabCreate) {
    tabCreate.classList.toggle("on", activePage === "create");
    tabCreate.setAttribute("aria-selected", activePage === "create" ? "true" : "false");
  }
  if (tabHistory) {
    tabHistory.classList.toggle("on", activePage === "history");
    tabHistory.setAttribute(
      "aria-selected",
      activePage === "history" ? "true" : "false"
    );
  }
  // Keep hash for refresh-friendly flip without losing JS state
  try {
    const want = activePage === "history" ? "#history" : "#create";
    if (location.hash !== want) {
      history.replaceState(null, "", want);
    }
  } catch {
    /* ignore */
  }
  if (activePage === "history") {
    // Re-bind infinite scroll if needed after becoming visible
    setupGalleryInfiniteScroll();
  }
}

function phaseLabel(phase) {
  if (!phase) return "";
  return PHASE_LABELS[phase] || String(phase).replace(/_/g, " ");
}

function formatBytes(n) {
  if (n == null || Number.isNaN(n)) return "";
  const u = ["B", "KB", "MB", "GB"];
  let v = Number(n);
  let i = 0;
  while (v >= 1024 && i < u.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v < 10 && i > 0 ? v.toFixed(1) : Math.round(v)} ${u[i]}`;
}

function setCancelActiveVisible(show, jobId) {
  const btn = $("btn-cancel-active");
  if (!btn) return;
  btn.hidden = !show;
  btn.disabled = !show;
  btn.dataset.jobId = jobId || "";
}

function updateJobStatusUI(job) {
  const box = $("job-status");
  if (!box) return;
  const phase = job?.phase || job?.status || "";
  const prog = job?.progress || {};
  const busy =
    job &&
    (job.status === "queued" ||
      job.status === "pulling" ||
      job.status === "running" ||
      phase === "cancelling");
  if (!job || !busy) {
    return;
  }
  box.hidden = false;
  setCancelActiveVisible(
    job.status === "pulling" ||
      job.status === "running" ||
      phase === "cancelling",
    job.id
  );

  $("status-phase").textContent = phaseLabel(phase) || job.status || "…";
  const bar = $("status-bar");
  const pctEl = $("status-pct");
  let pct = null;
  if (prog.current != null && prog.total != null && Number(prog.total) > 0) {
    pct = Math.max(0, Math.min(100, (100 * Number(prog.current)) / Number(prog.total)));
  }
  if (pct != null) {
    bar.classList.remove("indeterminate");
    bar.style.width = `${pct}%`;
    if (prog.unit === "byte") {
      pctEl.textContent = `${formatBytes(prog.current)} / ${formatBytes(prog.total)}`;
    } else if (prog.unit === "step") {
      pctEl.textContent = `${prog.current} / ${prog.total}`;
    } else {
      pctEl.textContent = `${Math.round(pct)}%`;
    }
  } else {
    bar.classList.add("indeterminate");
    bar.style.width = "35%";
    pctEl.textContent = "";
  }
  const detailParts = [];
  if (prog.message) detailParts.push(prog.message);
  if (job.meta?.profile) detailParts.push(job.meta.profile);
  if (job.meta?.seed != null) detailParts.push(`seed ${job.meta.seed}`);
  if (job.modality) detailParts.push(job.modality);
  if (job.id) detailParts.push(String(job.id).slice(0, 8));
  $("status-detail").textContent = detailParts.join(" · ");
}

function hideJobStatusIfIdle(snap) {
  const box = $("job-status");
  if (!box) return;
  const active = snap?.active_job;
  if (!active && !snap?.paused) {
    box.hidden = true;
    setCancelActiveVisible(false);
  }
}

function promptSnippet(job, n = 48) {
  const p = (job?.prompt || job?.meta?.positive || "").trim().replace(/\s+/g, " ");
  if (!p) return job?.modality || "job";
  return p.length > n ? p.slice(0, n - 1) + "…" : p;
}

function renderQueue(snap) {
  lastQueueSnap = snap;
  const list = $("queue-list");
  const empty = $("queue-empty");
  const banner = $("queue-banner");
  if (!list) return;

  const unloadBtn = $("btn-unload");
  if (unloadBtn) {
    unloadBtn.disabled = !!snap.busy;
    unloadBtn.title = snap.busy
      ? "Unload disabled while a job is running"
      : "Unload models from VRAM";
  }

  if (banner) {
    if (snap.paused) {
      banner.hidden = false;
      banner.textContent =
        "Paused at breakpoint — remove or move the break to resume.";
    } else {
      banner.hidden = true;
      banner.textContent = "";
    }
  }

  // Clear dynamic rows (keep empty placeholder node)
  [...list.querySelectorAll(".queue-row")].forEach((el) => el.remove());

  const rows = [];
  if (snap.active_job) {
    rows.push({ kind: "active", job: snap.active_job });
  }
  for (const ent of snap.entries || []) {
    rows.push(ent);
  }

  if (!rows.length) {
    if (empty) {
      empty.hidden = false;
      empty.textContent = "No jobs queued.";
    }
    hideJobStatusIfIdle(snap);
    return;
  }
  if (empty) empty.hidden = true;

  for (const ent of rows) {
    const row = document.createElement("div");
    row.className = "queue-row";
    if (ent.kind === "active") {
      row.classList.add("active");
      const job = ent.job;
      updateJobStatusUI(job);
      row.innerHTML = `
        <div class="queue-main">
          <div class="queue-title">
            <span class="queue-badge run">${
              job.phase === "cancelling" ? "stopping" : "running"
            }</span>
            ${escapeHtml(phaseLabel(job.phase || job.status))}
            · ${escapeHtml(job.modality || "")}
          </div>
          <div class="queue-sub">${escapeHtml(promptSnippet(job))}${
            job.progress?.message
              ? " · " + escapeHtml(job.progress.message)
              : ""
          }</div>
        </div>
        <div class="queue-actions">
          <button type="button" class="ghost icon-tiny danger-btn" data-act="cancel-active" title="Stop job">⏹</button>
        </div>`;
      const stopBtn = row.querySelector('[data-act="cancel-active"]');
      if (stopBtn) {
        stopBtn.onclick = () => cancelJobById(job.id);
      }
    } else if (ent.kind === "breakpoint") {
      row.classList.add("break");
      if (ent.active_pause) row.classList.add("active-pause");
      row.innerHTML = `
        <div class="queue-main">
          <div class="queue-title">
            <span class="queue-badge break">break</span>
            ${escapeHtml(ent.label || "Break")}
          </div>
          <div class="queue-sub">${
            ent.active_pause ? "Worker paused here" : "Stops queue until removed"
          }</div>
        </div>
        <div class="queue-actions">
          <button type="button" class="ghost icon-tiny" data-act="up" title="Move up">↑</button>
          <button type="button" class="ghost icon-tiny" data-act="down" title="Move down">↓</button>
          <button type="button" class="ghost icon-tiny" data-act="rm" title="Remove break">×</button>
        </div>`;
      wireQueueActions(row, ent.id);
    } else if (ent.kind === "job" && ent.job) {
      const job = ent.job;
      row.innerHTML = `
        <div class="queue-main">
          <div class="queue-title">
            <span class="queue-badge queued">queued</span>
            ${escapeHtml(job.modality || "")}${
              job.video_profile || job.meta?.profile
                ? " · " + escapeHtml(job.video_profile || job.meta.profile)
                : ""
            }
          </div>
          <div class="queue-sub">${escapeHtml(promptSnippet(job))}</div>
        </div>
        <div class="queue-actions">
          <button type="button" class="ghost icon-tiny" data-act="top" title="Elevate to top">⇈</button>
          <button type="button" class="ghost icon-tiny" data-act="up" title="Move up">↑</button>
          <button type="button" class="ghost icon-tiny" data-act="down" title="Move down">↓</button>
          <button type="button" class="ghost icon-tiny" data-act="rm" title="Cancel">×</button>
        </div>`;
      wireQueueActions(row, ent.id);
    } else {
      continue;
    }
    list.appendChild(row);
  }

  if (snap.active_job) {
    updateJobStatusUI(snap.active_job);
    $("job-status").hidden = false;
  } else if (snap.paused) {
    $("job-status").hidden = false;
    setCancelActiveVisible(false);
    $("status-phase").textContent = "Paused at breakpoint";
    $("status-pct").textContent = "";
    $("status-bar").classList.remove("indeterminate");
    $("status-bar").style.width = "0%";
    $("status-detail").textContent = "Remove or shift the break to continue.";
  } else {
    hideJobStatusIfIdle(snap);
  }
}

async function cancelJobById(jobId) {
  if (!jobId) return;
  try {
    setStatus("Stopping job…");
    const btn = $("btn-cancel-active");
    if (btn) btn.disabled = true;
    await api(`/v1/jobs/${encodeURIComponent(jobId)}/cancel`, { method: "POST" });
    setStatus(`Stop requested · ${String(jobId).slice(0, 8)}`);
    await refreshQueue();
  } catch (e) {
    setStatus(String(e.message || e), true);
  }
}

function escapeHtml(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function wireQueueActions(row, entryId) {
  row.querySelectorAll("button[data-act]").forEach((btn) => {
    btn.onclick = async () => {
      const act = btn.getAttribute("data-act");
      try {
        if (act === "rm") {
          await api(`/v1/queue/entries/${encodeURIComponent(entryId)}`, {
            method: "DELETE",
          });
        } else {
          await api("/v1/queue/reorder", {
            method: "POST",
            body: JSON.stringify({ id: entryId, action: act }),
          });
        }
        await refreshQueue();
      } catch (e) {
        setStatus(String(e.message || e), true);
      }
    };
  });
}

async function refreshQueue() {
  try {
    const snap = await api("/v1/queue");
    // Detect newly completed session jobs
    const activeId = snap.active_job_id;
    for (const id of [...sessionJobIds]) {
      if (id === activeId) continue;
      if (seenCompletedIds.has(id)) continue;
      // Check if still queued
      const stillQueued = (snap.entries || []).some(
        (e) => e.kind === "job" && e.id === id
      );
      if (stillQueued) continue;
      // Fetch job
      try {
        const job = await api(`/v1/jobs/${id}`);
        if (job.status === "completed") {
          seenCompletedIds.add(id);
          await onSessionJobCompleted(job);
        } else if (job.status === "failed" || job.status === "cancelled") {
          seenCompletedIds.add(id);
          if (job.status === "failed") {
            setStatus(job.error || "Job failed", true);
          } else {
            setStatus(`Cancelled · ${String(id).slice(0, 8)}`);
          }
        }
      } catch {
        /* ignore */
      }
    }
    renderQueue(snap);
  } catch (e) {
    // Don't spam status on poll errors
    console.warn("queue poll", e);
  }
}

async function onSessionJobCompleted(job) {
  const outs = job.outputs || [];
  const out =
    outs.find((o) => isVideoOutput(o)) ||
    outs.find((o) => o.kind === "image") ||
    outs[0];
  if (out?.url) {
    const kind =
      out.kind ||
      (/\.(mp4|webm|mov|mkv)$/i.test(out.filename || out.url)
        ? "video"
        : "image");
    showResultMedia(out.url, kind);
  }
  const sz =
    job.meta?.width && job.meta?.height
      ? ` · ${job.meta.width}×${job.meta.height}`
      : "";
  const prof = job.meta?.profile ? ` · ${job.meta.profile}` : "";
  setStatus(
    `Done · seed=${job.seed ?? job.meta?.seed ?? "?"}${sz}${prof} · job ${String(job.id).slice(0, 8)}`
  );
  updateJobStatusUI({ ...job, status: "completed", phase: "done" });
  await loadGallery({ reset: true });
  const fresh = galleryItems.find((g) => g.id === job.id);
  if (fresh) selectGalleryItem(fresh);
}

function startQueuePolling() {
  if (queuePollTimer) return;
  queuePollTimer = setInterval(() => {
    refreshQueue().catch(() => {});
  }, 1000);
}

function modalityInfo(id) {
  return (
    (capabilities.modalities || []).find((m) => m.id === id) || {
      id,
      implemented: id === "txt2img",
    }
  );
}

function videoPackDefaults() {
  const p = (capabilities.video_packs || []).find((x) => x.id === "sample");
  if (p?.defaults) return p.defaults;
  return { width: 640, height: 640, length: 81, fps: 16 };
}

function currentPreset() {
  return presets.find((x) => x.id === selected) || null;
}

function effectCompatible(eff, family) {
  if (!eff.base_families || !eff.base_families.length) return true;
  return eff.base_families
    .map((f) => f.toLowerCase())
    .includes((family || "").toLowerCase());
}

function effectsModality() {
  if (modality === "txt2img") return "txt2img";
  if (modality === "txt2vid") return "txt2vid";
  if (modality === "img2vid") return "img2vid";
  if (isVid2Vid() && (vid2vidOp === "continue" || vid2vidOp === "finalize")) {
    if (vid2vidOp === "continue") return "img2vid";
    const sm = sourceVideo?.srcModality || sourceVideo?.meta?.modality || "";
    return sm === "txt2vid" ? "txt2vid" : "img2vid";
  }
  return null;
}

function effectAppliesToCurrent(eff) {
  const mod = effectsModality();
  if (!mod) return false;
  const mods = eff.modalities || ["txt2img"];
  return mods.map((m) => String(m).toLowerCase()).includes(mod);
}

function snap16(n) {
  n = Math.max(16, Math.round(Number(n) || 0));
  return n - (n % 16);
}

function fitVideoSize(w, h, maxSide = 832) {
  w = Number(w);
  h = Number(h);
  if (!w || !h) return null;
  const longest = Math.max(w, h);
  const scale = longest > maxSide ? maxSide / longest : 1;
  return { width: snap16(w * scale), height: snap16(h * scale) };
}

async function refreshHealth() {
  const el = $("health");
  try {
    const h = await api("/v1/health");
    const impl = (h.modalities?.implemented || []).join(", ") || "txt2img";
    const feat = h.features || {};
    const cont =
      feat.continue_method_default === "flf_bridge"
        ? "continue:FLF"
        : feat.continue_cfr_xfade
          ? "continue:legacy"
          : "continue:?";
    if (h.comfy_reachable) {
      el.textContent = `OK · Comfy · ${h.presets} presets · ${cont} · live: ${impl}`;
      el.className = "health ok";
    } else {
      el.textContent = `API up · Comfy offline · ${cont} · live: ${impl}`;
      el.className = "health bad";
    }
    // Warn if browser still talks to a pre-FLF server
    if (!feat.continue_motion_prefix || feat.continue_method_default !== "flf_bridge") {
      el.title =
        "Server is missing current continue features — restart simplecomfy serve";
      el.className = "health bad";
    } else {
      el.title = `simpleComfy ${h.version || ""} · FLF bridge continue enabled`;
    }
  } catch {
    el.textContent = "API unreachable";
    el.className = "health bad";
  }
}

function renderModes() {
  const box = $("modes");
  box.innerHTML = "";
  for (const m of MODES) {
    const info = modalityInfo(m.id);
    // vid2vid is UI-level even if API lists smooth/continue separately
    const implemented =
      m.id === "vid2vid"
        ? true
        : info.implemented !== false || m.id === "txt2img";
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "chip" + (modality === m.id ? " on" : "");
    btn.innerHTML =
      m.label + (!implemented ? '<span class="soon">soon</span>' : "");
    btn.title = info.note || m.label;
    btn.onclick = () => {
      modality = m.id;
      applyModeUI();
      renderModes();
      updateSizeHint();
      closeResetMenus();
    };
    box.appendChild(btn);
  }
  applyModeUI();
}

function videoProfileMeta() {
  const pack = (capabilities.video_packs || []).find((p) => p.id === "sample");
  const list = pack?.profiles || [];
  const found = list.find((p) => p.id === videoProfile);
  return found || { id: videoProfile, label: videoProfile, description: "" };
}

function renderVid2VidOps() {
  const box = $("vid2vid-ops");
  if (!box) return;
  box.innerHTML = "";
  for (const op of VID2VID_OPS) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "chip" + (vid2vidOp === op.id ? " on" : "");
    btn.textContent = op.label;
    btn.title = op.description;
    btn.onclick = () => {
      vid2vidOp = op.id;
      if (op.id === "finalize") videoProfile = "final";
      renderVid2VidOps();
      applyModeUI();
    };
    box.appendChild(btn);
  }
}

function renderSmoothFactors() {
  const box = $("smooth-factors");
  if (!box) return;
  box.innerHTML = "";
  for (const f of [2, 4]) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "chip" + (smoothFactor === f ? " on" : "");
    btn.textContent = `${f}× frames`;
    btn.title = `Interpolate to ~${f}× frame rate`;
    btn.onclick = () => {
      smoothFactor = f;
      renderSmoothFactors();
    };
    box.appendChild(btn);
  }
}

function renderVideoProfiles() {
  const box = $("video-profiles");
  if (!box) return;
  box.innerHTML = "";
  const pack = (capabilities.video_packs || []).find((p) => p.id === "sample");
  const profiles =
    pack?.profiles?.length
      ? pack.profiles
      : [
          {
            id: "draft",
            label: "Draft",
            description: "4-step draft",
          },
          {
            id: "final",
            label: "Final",
            description: "Full sampler",
          },
          { id: "fast", label: "Fast", description: "4-step draft" },
          { id: "quality", label: "Quality", description: "Full sampler" },
        ];
  if (pack?.default_profile && !profiles.find((p) => p.id === videoProfile)) {
    videoProfile = pack.default_profile;
  }
  // Finalize locks to Final
  const locked = isVid2Vid() && vid2vidOp === "finalize";
  if (locked) videoProfile = "final";
  for (const p of profiles) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "chip" + (videoProfile === p.id ? " on" : "");
    btn.disabled = locked && p.id !== "final";
    const draftish = p.id === "draft" || p.id === "fast";
    const finalish = p.id === "final" || p.id === "quality";
    btn.textContent = p.label + (p.ready === false ? " ·" : "");
    btn.title =
      (p.description || p.id) +
      (p.steps != null ? ` · ${p.steps} steps` : "") +
      (draftish ? " · draft speed" : "") +
      (finalish ? " · final quality" : "") +
      (locked && p.id !== "final" ? " · locked off for Finalize" : "") +
      (p.ready === false ? " · needs pull" : "");
    btn.onclick = () => {
      if (locked) return;
      videoProfile = p.id;
      renderVideoProfiles();
      applyModeUI();
    };
    box.appendChild(btn);
  }
}

function applyModeUI() {
  const info = modalityInfo(modality);
  const needsImageSource = modality === "img2vid";
  const needsVideoSource = isVid2Vid();
  const needsSource = needsImageSource || needsVideoSource;
  const showProfiles =
    modality === "txt2vid" ||
    modality === "img2vid" ||
    (isVid2Vid() && (vid2vidOp === "continue" || vid2vidOp === "finalize"));
  const showSmoothFactor = isVid2Vid() && vid2vidOp === "smooth";
  const showPrompt =
    modality === "txt2img" ||
    modality === "txt2vid" ||
    modality === "img2vid" ||
    (isVid2Vid() && (vid2vidOp === "continue" || vid2vidOp === "finalize"));
  const showSeedSize =
    modality === "txt2img" ||
    modality === "txt2vid" ||
    modality === "img2vid" ||
    (isVid2Vid() && (vid2vidOp === "continue" || vid2vidOp === "finalize"));
  const showContinueSecs = isVid2Vid() && vid2vidOp === "continue";
  const showContinueOverlap = showContinueSecs;
  const showContinueContext = showContinueSecs;
  const showContinueMethod = showContinueSecs;
  const showVideoSecs =
    modality === "txt2vid" ||
    modality === "img2vid" ||
    (isVid2Vid() && vid2vidOp === "finalize");
  const showContextWindows = showVideoSecs;

  $("panel-preset").hidden = modality !== "txt2img";
  const showEffects =
    modality === "txt2img" ||
    modality === "txt2vid" ||
    modality === "img2vid" ||
    (isVid2Vid() && (vid2vidOp === "continue" || vid2vidOp === "finalize"));
  $("panel-effects").hidden = !showEffects;
  if (showEffects) {
    const elHint = $("effects-hint");
    if (modality === "txt2img") {
      if (elHint) elHint.textContent = "(optional · multi-select)";
    } else {
      if (elHint) elHint.textContent = "(optional · motion / act LoRAs)";
    }
    renderEffects();
  }
  $("panel-source").hidden = !needsSource;
  $("panel-vid2vid-ops").hidden = !isVid2Vid();
  $("panel-video-profile").hidden = !showProfiles;
  $("panel-smooth-factor").hidden = !showSmoothFactor;
  $("panel-prompt").hidden = !showPrompt;
  $("panel-negative").hidden = !showPrompt;
  $("field-seed").hidden = !showSeedSize;
  $("field-width").hidden = !showSeedSize;
  $("field-height").hidden = !showSeedSize;
  $("field-continue-secs").hidden = !showContinueSecs;
  if ($("field-continue-method")) $("field-continue-method").hidden = !showContinueMethod;
  if ($("field-continue-context")) $("field-continue-context").hidden = !showContinueContext;
  if ($("field-continue-overlap")) $("field-continue-overlap").hidden = !showContinueOverlap;
  if ($("field-video-secs")) $("field-video-secs").hidden = !showVideoSecs;
  if ($("field-context-windows")) $("field-context-windows").hidden = !showContextWindows;

  if (showContinueMethod) renderContinueMethodChips();
  if (showContinueContext) renderContextFramePresets();
  if (showContinueOverlap) renderOverlapPresets();
  if (showVideoSecs) updateVideoSecsHint();
  if (showContextWindows) {
    renderContextModeChips();
    renderContextOverlapPresets();
    updateContextWindowsHint();
  }

  if (isVid2Vid()) {
    renderVid2VidOps();
    if (showSmoothFactor) renderSmoothFactors();
  }
  if (showProfiles) renderVideoProfiles();

  // Source panel labels
  const srcLabel = $("source-label");
  const srcHint = $("source-hint");
  const srcActionsImg = $("source-actions-image");
  if (needsVideoSource) {
    srcLabel.textContent = "Source video";
    if (srcHint) srcHint.textContent = "Use in → Video → Video from a History clip";
    if (srcActionsImg) srcActionsImg.hidden = true;
  } else if (needsImageSource) {
    srcLabel.innerHTML =
      'Source image <span class="hint">(Image → Video)</span>';
    if (srcHint) srcHint.textContent = "or Use in → from a History still";
    if (srcActionsImg) srcActionsImg.hidden = false;
  }

  const gen = $("btn-gen");
  if (isVid2Vid()) {
    const op = VID2VID_OPS.find((o) => o.id === vid2vidOp);
    gen.textContent =
      vid2vidOp === "smooth"
        ? `Smooth ${smoothFactor}×`
        : vid2vidOp === "continue"
          ? "Continue"
          : "Generate Final";
    gen.title = op?.description || "";
  } else {
    gen.textContent = "Generate";
    gen.title = info.note || "";
  }

  const pl = $("prompt-label");
  const ph = $("prompt");
  if (modality === "img2vid") {
    pl.textContent = "Motion prompt";
    ph.placeholder = "Optional: how the image should move…";
  } else if (modality === "txt2vid") {
    pl.textContent = "Video prompt";
    ph.placeholder = "Describe the video…";
  } else if (isVid2Vid() && vid2vidOp === "continue") {
    pl.textContent = "Motion prompt";
    ph.placeholder = "Optional override — empty inherits from source clip…";
  } else if (isVid2Vid() && vid2vidOp === "finalize") {
    pl.textContent = "Prompt";
    ph.placeholder = "Recipe prompt for the Final pass…";
  } else {
    pl.textContent = "Prompt";
    ph.placeholder = "Describe the image…";
  }

  const mm = $("mode-meta");
  const vp = videoProfileMeta();
  if (modality === "txt2img") {
    mm.textContent = "Pick a preset, optional effects, then generate.";
  } else if (modality === "txt2vid") {
    mm.textContent = `the video pack T2V · profile “${vp.label}”${
      vp.description ? " — " + vp.description : ""
    }`;
  } else if (modality === "img2vid") {
    mm.textContent = `the video pack I2V · profile “${vp.label}” · source res by default${
      vp.description ? " — " + vp.description : ""
    }`;
  } else if (isVid2Vid()) {
    if (vid2vidOp === "smooth") {
      mm.textContent = `Smooth · ${smoothFactor}× temporal upsample (ffmpeg) · pick a History video`;
    } else if (vid2vidOp === "continue") {
      mm.textContent =
        continueMethod === "flf_bridge"
          ? `Continue · motion prefix → ${vp.label} chunk → FLF bridge → double xfade`
          : `Continue · motion prefix → ${vp.label} chunk → seam xfade (no FLF)`;
    } else {
      mm.textContent =
        "Final · re-run source clip’s recipe at full quality · pick a History video";
    }
  }

  const def = sizeDefaultsForMode();
  $("width").placeholder = def ? String(def.width) : "default";
  $("height").placeholder = def ? String(def.height) : "default";

  renderSource();
  updateSizeHint();
}

function sizeDefaultsForMode() {
  if (modality === "txt2img") {
    const p = currentPreset();
    if (p?.default_size) return p.default_size;
    return { width: 1024, height: 1024 };
  }
  return videoPackDefaults();
}

function sourceVideoSize() {
  if (!sourceImage) return null;
  if (sourceImage.videoWidth && sourceImage.videoHeight) {
    return { width: sourceImage.videoWidth, height: sourceImage.videoHeight };
  }
  if (sourceImage.width && sourceImage.height) {
    return fitVideoSize(sourceImage.width, sourceImage.height);
  }
  return null;
}

function updateSizeHint() {
  const el = $("size-hint");
  if (!el) return;
  if (isVid2Vid() && vid2vidOp === "smooth") {
    el.textContent = "Smooth keeps the source resolution; only frame rate changes.";
    return;
  }
  if (isVid2Vid() && vid2vidOp === "continue") {
    const secs = $("continue-secs")?.value || "5";
    const ctx = $("continue-context")?.value ?? "12";
    const ov = $("continue-overlap")?.value ?? "0.75";
    el.textContent =
      `Extends ~${secs}s · motion context ${ctx}f` +
      (Number(ctx) <= 1 ? " (still only)" : " + warmup strip") +
      ` · seam blend ${ov}s` +
      (Number(ov) <= 0 ? " (hard cut)" : " (CFR xfade)");
    return;
  }
  const w = $("width").value;
  const h = $("height").value;
  const def = sizeDefaultsForMode();
  const src = sourceVideoSize();
  if (modality === "img2vid" || (isVid2Vid() && vid2vidOp === "finalize")) {
    if (w && h) {
      el.textContent = `Using ${w}×${h}` + (src ? ` · source fit ${src.width}×${src.height}` : "");
    } else if (src) {
      el.textContent = `Empty fields → source resolution ${src.width}×${src.height} (aspect preserved, max side 832)`;
    } else if (sourceVideo?.width && sourceVideo?.height) {
      el.textContent = `Empty fields → clip size ${sourceVideo.width}×${sourceVideo.height}`;
    } else {
      el.textContent = `Empty fields → pack default ${def.width}×${def.height}`;
    }
  } else if (modality === "txt2vid") {
    el.textContent = w && h
      ? `Using ${w}×${h}`
      : `Empty fields → pack default ${def.width}×${def.height}`;
  } else {
    const p = currentPreset();
    const pd = p?.default_size || def;
    el.textContent = w && h
      ? `Using ${w}×${h}`
      : `Empty fields → preset default ${pd.width}×${pd.height}`;
  }
}

function renderPresets() {
  const box = $("presets");
  box.innerHTML = "";
  for (const p of presets) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "chip" + (selected === p.id ? " on" : "");
    btn.textContent = p.label + (p.ready ? "" : " ·");
    btn.title = p.ready
      ? p.description || p.id
      : `Models not local yet (~${p.size_gb_approx ?? "?"} GB)`;
    btn.onclick = () => {
      selected = p.id;
      const fam = p.base_family || "";
      selectedEffects = new Set(
        (p.default_effects || []).filter((id) => {
          const e = allEffects.find((x) => x.id === id);
          return e && effectCompatible(e, fam);
        })
      );
      renderPresets();
      renderEffects();
      updateMeta();
      applyModeUI();
    };
    box.appendChild(btn);
  }
  if (!selected && presets.length) {
    selected = presets[0].id;
    selectedEffects = new Set(presets[0].default_effects || []);
    renderPresets();
    return;
  }
  updateMeta();
}

function renderEffects() {
  const box = $("effects");
  if (!box) return;
  box.innerHTML = "";
  const mod = effectsModality();
  const p = currentPreset();
  const family = p?.base_family || "";
  const suggested = new Set(p?.suggested_effects || []);
  let visible = allEffects.filter((e) => effectAppliesToCurrent(e));
  if (mod === "txt2img") {
    visible = visible.filter((e) => effectCompatible(e, family));
  }
  for (const id of [...selectedEffects]) {
    if (!visible.some((e) => e.id === id)) selectedEffects.delete(id);
  }
  const sorted = [...visible].sort((a, b) => {
    const as = suggested.has(a.id) ? 0 : 1;
    const bs = suggested.has(b.id) ? 0 : 1;
    if (as !== bs) return as - bs;
    // Group bounce family together
    const ag = a.exclusive_group || "";
    const bg = b.exclusive_group || "";
    if (ag !== bg) return ag.localeCompare(bg);
    return a.label.localeCompare(b.label);
  });
  if (!sorted.length) {
    box.innerHTML =
      mod === "txt2img"
        ? '<span class="meta">No effects for this family.</span>'
        : '<span class="meta">No video effects for this mode.</span>';
    updateEffectMeta();
    return;
  }
  for (const e of sorted) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "chip" + (selectedEffects.has(e.id) ? " on" : "");
    const tag = e.needs_lora ? (e.ready ? "" : " ·") : "";
    const groupTag = e.exclusive_group
      ? `<span class="tag">${e.exclusive_group}</span>`
      : "";
    btn.innerHTML =
      e.label +
      tag +
      (suggested.has(e.id) ? '<span class="tag">★</span>' : "") +
      groupTag +
      (!e.needs_lora ? '<span class="tag">txt</span>' : "");
    btn.title =
      (e.description || e.id) +
      (e.exclusive_group ? ` · group “${e.exclusive_group}” (one at a time)` : "") +
      (e.needs_lora ? (e.ready ? " · LoRA ready" : " · pulls on generate") : " · prompt-only");
    btn.onclick = () => {
      if (selectedEffects.has(e.id)) {
        selectedEffects.delete(e.id);
      } else {
        // Exclusive group: deselect others in same group
        if (e.exclusive_group) {
          for (const other of visible) {
            if (
              other.exclusive_group === e.exclusive_group &&
              other.id !== e.id
            ) {
              selectedEffects.delete(other.id);
            }
          }
        }
        selectedEffects.add(e.id);
      }
      renderEffects();
      updateEffectMeta();
    };
    box.appendChild(btn);
  }
  updateEffectMeta();
}

function updateMeta() {
  const p = currentPreset();
  const el = $("preset-meta");
  if (!p) {
    el.textContent = "No presets found.";
    return;
  }
  const ready = p.ready ? "ready" : "needs pull";
  const size = p.size_gb_approx != null ? ` · ~${p.size_gb_approx} GB` : "";
  el.textContent = `${p.id} · ${p.base_family} · ${ready}${size}${
    p.description ? " — " + p.description : ""
  }`;
}

function updateEffectMeta() {
  const el = $("effect-meta");
  if (!el) return;
  const ids = [...selectedEffects];
  if (!ids.length) {
    el.textContent =
      effectsModality() === "txt2img"
        ? "No effects selected."
        : "No video effects — profile LoRAs only.";
    return;
  }
  el.textContent =
    "Active: " +
    ids
      .map((id) => allEffects.find((x) => x.id === id)?.label || id)
      .join(", ");
}

function renderSource() {
  const box = $("source-preview");
  const meta = $("source-meta");
  if (isVid2Vid()) {
    if (!sourceVideo) {
      box.className = "source-preview empty";
      box.textContent = "Pick a video via History → Use in → Video → Video";
      meta.textContent = "";
      return;
    }
    box.className = "source-preview";
    box.innerHTML = "";
    const v = document.createElement("video");
    v.src = sourceVideo.url;
    v.muted = true;
    v.loop = true;
    v.playsInline = true;
    v.autoplay = true;
    v.controls = false;
    box.appendChild(v);
    const bits = [
      `clip ${sourceVideo.jobId.slice(0, 8)}… / ${sourceVideo.filename}`,
    ];
    if (sourceVideo.width && sourceVideo.height) {
      bits.push(`${sourceVideo.width}×${sourceVideo.height}`);
    }
    if (sourceVideo.profile) bits.push(sourceVideo.profile);
    if (sourceVideo.srcModality) bits.push(sourceVideo.srcModality);
    meta.textContent = bits.join(" · ");
    return;
  }

  if (!sourceImage) {
    box.className = "source-preview empty";
    box.textContent = "Pick from gallery or disk";
    meta.textContent = "";
    return;
  }
  box.className = "source-preview";
  box.innerHTML = "";
  const img = document.createElement("img");
  img.src = sourceImage.url;
  img.alt = "source";
  box.appendChild(img);
  const bits = [];
  if (sourceImage.type === "gallery") {
    bits.push(`gallery ${sourceImage.jobId.slice(0, 8)}… / ${sourceImage.filename}`);
  } else {
    bits.push(`disk · ${sourceImage.input}`);
  }
  if (sourceImage.width && sourceImage.height) {
    bits.push(`${sourceImage.width}×${sourceImage.height}`);
  }
  const fit = sourceVideoSize();
  if (fit) bits.push(`video ${fit.width}×${fit.height}`);
  meta.textContent = bits.join(" · ");
}

function setSourceVideoFromGallery(item) {
  const vOut = firstVideoOf(item);
  if (!vOut?.url) {
    setStatus("No video on this item.", true);
    return false;
  }
  const m = item.meta || {};
  sourceVideo = {
    jobId: item.id,
    filename: vOut.filename || "00.mp4",
    url: vOut.url,
    width: item.width || m.width || null,
    height: item.height || m.height || null,
    profile: item.profile || m.profile || null,
    srcModality: item.modality || m.modality || null,
    meta: m,
    prompt: item.prompt || item.positive || m.positive || "",
    negative: item.negative_prompt || item.negative || m.negative || "",
    seed: item.seed != null ? item.seed : m.seed,
  };
  // Prefill form from clip recipe when useful
  if (sourceVideo.prompt && !$("prompt").value.trim()) {
    $("prompt").value = sourceVideo.prompt;
  }
  if (sourceVideo.negative && !$("negative").value.trim()) {
    $("negative").value = sourceVideo.negative;
  }
  if (sourceVideo.seed != null && sourceVideo.seed !== "" && $("seed").value === "-1") {
    $("seed").value = String(sourceVideo.seed);
  }
  if (sourceVideo.width) $("width").value = String(sourceVideo.width);
  if (sourceVideo.height) $("height").value = String(sourceVideo.height);
  if (sourceVideo.profile && vid2vidOp !== "finalize") {
    videoProfile = sourceVideo.profile;
  }
  if (vid2vidOp === "finalize") videoProfile = "final";
  renderSource();
  updateSizeHint();
  return true;
}

function setSourceFromGallery(item, imageOut) {
  const w = item.width || null;
  const h = item.height || null;
  // Prefer natural image dims when we can load them client-side later
  sourceImage = {
    type: "gallery",
    jobId: item.id,
    filename: imageOut.filename,
    url: imageOut.url,
    width: w,
    height: h,
    videoWidth: null,
    videoHeight: null,
  };
  // Probe actual pixels via Image()
  const probe = new Image();
  probe.onload = () => {
    sourceImage.width = probe.naturalWidth;
    sourceImage.height = probe.naturalHeight;
    const fit = fitVideoSize(probe.naturalWidth, probe.naturalHeight);
    if (fit) {
      sourceImage.videoWidth = fit.width;
      sourceImage.videoHeight = fit.height;
    }
    renderSource();
    if (modality === "img2vid") {
      // Always prefer source res when picking for img2vid (user can reset)
      $("width").value = String(fit?.width || probe.naturalWidth);
      $("height").value = String(fit?.height || probe.naturalHeight);
      updateSizeHint();
    }
  };
  probe.src = imageOut.url;
  renderSource();
  if (modality === "img2vid" && w && h) {
    const fit = fitVideoSize(w, h);
    if (fit) {
      sourceImage.videoWidth = fit.width;
      sourceImage.videoHeight = fit.height;
      $("width").value = String(fit.width);
      $("height").value = String(fit.height);
    }
  }
  updateSizeHint();
}

async function uploadSourceFile(file) {
  setStatus(`Uploading ${file.name}…`);
  const fd = new FormData();
  fd.append("file", file);
  const res = await api("/v1/inputs", { method: "POST", body: fd });
  sourceImage = {
    type: "upload",
    input: res.filename || res.comfy_name,
    url: res.url,
    width: res.width,
    height: res.height,
    videoWidth: res.video_width,
    videoHeight: res.video_height,
  };
  renderSource();
  if (modality === "img2vid") {
    const w = res.video_width || res.width;
    const h = res.video_height || res.height;
    if (w && h) {
      $("width").value = String(w);
      $("height").value = String(h);
    }
  }
  updateSizeHint();
  setStatus(
    `Source ready · ${res.width || "?"}×${res.height || "?"}` +
      (res.video_width ? ` → video ${res.video_width}×${res.video_height}` : "")
  );
}

/* ---------- size reset menus ---------- */
function closeResetMenus() {
  for (const id of ["menu-reset-width", "menu-reset-height"]) {
    const m = $(id);
    if (m) m.hidden = true;
  }
}

function buildResetMenu(menuEl, axis) {
  menuEl.innerHTML = "";
  const note = document.createElement("div");
  note.className = "menu-note";
  note.textContent = "Set " + axis;
  menuEl.appendChild(note);

  const add = (label, disabled, fn) => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = label;
    b.disabled = !!disabled;
    b.onclick = (e) => {
      e.stopPropagation();
      fn();
      closeResetMenus();
      updateSizeHint();
    };
    menuEl.appendChild(b);
  };

  const src = sourceVideoSize();
  const rawW = sourceImage?.width;
  const rawH = sourceImage?.height;
  const pack = videoPackDefaults();
  const preset = currentPreset()?.default_size;

  if (modality === "img2vid") {
    add(
      src
        ? `Source ${src.width}×${src.height}`
        : "Source (set an image first)",
      !src,
      () => {
        if (!src) return;
        $("width").value = String(src.width);
        $("height").value = String(src.height);
      }
    );
    if (rawW && rawH && src && (rawW !== src.width || rawH !== src.height)) {
      add(`Source native ${rawW}×${rawH} (no cap)`, false, () => {
        $("width").value = String(snap16(rawW));
        $("height").value = String(snap16(rawH));
      });
    }
    add(`Pack default ${pack.width}×${pack.height}`, false, () => {
      $("width").value = String(pack.width);
      $("height").value = String(pack.height);
    });
  } else if (modality === "txt2vid") {
    add(`Pack default ${pack.width}×${pack.height}`, false, () => {
      $("width").value = String(pack.width);
      $("height").value = String(pack.height);
    });
  } else {
    const pd = preset || { width: 1024, height: 1024 };
    add(`Preset default ${pd.width}×${pd.height}`, false, () => {
      $("width").value = String(pd.width);
      $("height").value = String(pd.height);
    });
  }

  add("Clear (use server default)", false, () => {
    $("width").value = "";
    $("height").value = "";
  });
}

function toggleResetMenu(which) {
  const menu = $(which === "width" ? "menu-reset-width" : "menu-reset-height");
  const other = $(which === "width" ? "menu-reset-height" : "menu-reset-width");
  if (other) other.hidden = true;
  if (!menu) return;
  if (menu.hidden) {
    buildResetMenu(menu, which);
    menu.hidden = false;
  } else {
    menu.hidden = true;
  }
}

/* ---------- gallery ---------- */
function fmtTime(iso) {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

function isVideoUrl(url, kind) {
  if (kind === "video") return true;
  if (!url) return false;
  return /\.(mp4|webm|mov|mkv)(\?|$)/i.test(url);
}

function isVideoOutput(o) {
  if (!o) return false;
  return isVideoUrl(o.url || o.filename || "", o.kind);
}

function mediaEl(url, kind, opts = {}) {
  const video = isVideoUrl(url, kind);
  if (video) {
    const v = document.createElement("video");
    v.src = url;
    v.muted = opts.muted !== false;
    v.loop = opts.loop !== false;
    v.playsInline = true;
    v.setAttribute("playsinline", "");
    v.preload = opts.preload || "metadata";
    if (opts.controls) v.controls = true;
    if (opts.autoplay) {
      v.autoplay = true;
      v.play?.().catch(() => {});
    }
    return v;
  }
  const img = document.createElement("img");
  img.src = url;
  img.alt = opts.alt || "";
  img.loading = opts.loading || "lazy";
  return img;
}

function teardownGalleryMedia(root) {
  if (!root) return;
  root.querySelectorAll("video").forEach((v) => {
    try {
      v.pause();
      v.removeAttribute("src");
      v.load();
    } catch {
      /* ignore */
    }
  });
}

function ensureGalleryThumbObserver() {
  if (galleryThumbObserver) return galleryThumbObserver;
  const root = $("gallery-scroll");
  galleryThumbObserver = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (!e.isIntersecting) continue;
        const v = e.target;
        if (!(v instanceof HTMLVideoElement)) continue;
        if (v.dataset.thumbLoaded === "1") continue;
        // Load a single frame for the poster — never autoplay a grid of clips
        v.dataset.thumbLoaded = "1";
        v.preload = "metadata";
        const src = v.dataset.src;
        if (src && !v.src) v.src = src;
        const onMeta = () => {
          try {
            // Seek slightly so browsers paint a frame instead of black
            if (v.readyState >= 1 && v.duration && !Number.isNaN(v.duration)) {
              v.currentTime = Math.min(0.05, v.duration * 0.01);
            }
          } catch {
            /* ignore */
          }
          v.removeEventListener("loadedmetadata", onMeta);
        };
        v.addEventListener("loadedmetadata", onMeta);
        // If metadata already cached
        if (v.readyState >= 1) onMeta();
        galleryThumbObserver.unobserve(v);
      }
    },
    { root: root || null, rootMargin: "80px", threshold: 0.01 }
  );
  return galleryThumbObserver;
}

function galleryCardEl(item) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.dataset.jobId = item.id;
  const tKind =
    item.thumb_kind || (item.has_video && !item.has_image ? "video" : "image");
  btn.className =
    "g-card" +
    (selectedGallery?.id === item.id ? " on" : "") +
    (item.has_video || tKind === "video" ? " is-video" : "");

  if (item.thumb_url) {
    if (tKind === "video" || isVideoUrl(item.thumb_url, tKind)) {
      // Do NOT set src + autoplay for every card — that OOM/crashes on fast scroll.
      const v = document.createElement("video");
      v.muted = true;
      v.playsInline = true;
      v.setAttribute("playsinline", "");
      v.preload = "none";
      v.controls = false;
      v.dataset.src = item.thumb_url;
      v.className = "g-thumb-video";
      btn.appendChild(v);
      ensureGalleryThumbObserver().observe(v);
    } else {
      const img = document.createElement("img");
      img.src = item.thumb_url;
      img.alt = "";
      img.loading = "lazy";
      img.decoding = "async";
      btn.appendChild(img);
    }
  } else {
    const ph = document.createElement("span");
    ph.className = "g-thumb-empty";
    ph.textContent = "·";
    btn.appendChild(ph);
  }

  const badge = document.createElement("span");
  badge.className = "g-badge";
  const mod = item.modality || "img";
  const shortMod = {
    txt2img: "img",
    txt2vid: "t2v",
    img2vid: "i2v",
    smooth: "smooth",
    continue: "cont",
    vid2vid: "v2v",
  }[mod] || mod;
  badge.textContent =
    (mod === "txt2vid" ||
    mod === "img2vid" ||
    mod === "smooth" ||
    mod === "continue" ||
    tKind === "video"
      ? "▶ "
      : "") + shortMod;
  btn.appendChild(badge);
  btn.title = [
    item.preset || item.modality || "output",
    item.prompt ? item.prompt.slice(0, 80) : "(no prompt)",
    fmtTime(item.created_at),
  ].join("\n");
  btn.onclick = () => selectGalleryItem(item);
  return btn;
}

function updateGallerySelection() {
  const box = $("gallery");
  if (!box) return;
  const sel = selectedGallery?.id || "";
  box.querySelectorAll(".g-card").forEach((el) => {
    el.classList.toggle("on", el.dataset.jobId === sel);
  });
}

/**
 * Render gallery cards.
 * @param {{ reset?: boolean }} [opts]
 *  reset=true rebuilds the grid; otherwise only appends new items (fast scroll safe).
 */
function renderGallery({ reset = false } = {}) {
  const box = $("gallery");
  const empty = $("gallery-empty");
  if (!box) return;
  if (empty) empty.hidden = galleryItems.length > 0;

  if (reset || galleryRenderedCount > galleryItems.length) {
    if (galleryThumbObserver) {
      galleryThumbObserver.disconnect();
      galleryThumbObserver = null;
    }
    teardownGalleryMedia(box);
    box.innerHTML = "";
    galleryRenderedCount = 0;
  }

  const start = galleryRenderedCount;
  const frag = document.createDocumentFragment();
  for (let i = start; i < galleryItems.length; i++) {
    frag.appendChild(galleryCardEl(galleryItems[i]));
  }
  box.appendChild(frag);
  galleryRenderedCount = galleryItems.length;
  updateGallerySelection();
}

function selectGalleryItem(item) {
  selectedGallery = item;
  updateGallerySelection();
  showDetail(item);
}

function showDetail(item) {
  const panel = $("detail-panel");
  const placeholder = $("detail-placeholder");
  panel.hidden = false;
  if (placeholder) placeholder.hidden = true;
  const media = $("detail-media");
  media.innerHTML = "";
  const outs = item.outputs || [];
  const first =
    outs.find((o) => isVideoOutput(o)) ||
    outs.find((o) => o.kind === "image") ||
    outs[0];
  if (first?.url) {
    const el = mediaEl(first.url, first.kind, {
      controls: true,
      autoplay: isVideoOutput(first),
      loop: true,
      muted: true,
      preload: "auto",
      alt: "output",
    });
    if (el.tagName === "VIDEO") el.controls = true;
    media.appendChild(el);
  }

  const loras = (item.loras || [])
    .map((l) =>
      typeof l === "string" ? l : `${l.name || l.filename || "?"}@${l.weight ?? ""}`
    )
    .join(", ");
  const effects = Array.isArray(item.effects)
    ? item.effects
        .map((e) => (typeof e === "string" ? e : e.id))
        .filter(Boolean)
        .join(", ")
    : "";

  const rows = [
    ["When", fmtTime(item.created_at)],
    ["Mode", item.modality || "txt2img"],
    ["Profile", item.profile || item.meta?.profile || null],
    ["Preset", item.preset || "—"],
    ["Checkpoint", item.checkpoint || "—"],
    ["LoRAs", loras || "—"],
    ["Effects", effects || "—"],
    ["Seed", item.seed != null ? String(item.seed) : "—"],
    [
      "Size",
      item.width && item.height ? `${item.width}×${item.height}` : "—",
    ],
    [
      "Sampler",
      item.sampler_name
        ? `${item.sampler_name} / ${item.scheduler || "?"} · ${item.steps ?? "?"} steps · cfg ${item.cfg ?? "?"}`
        : "—",
    ],
    [
      "Video",
      item.fps ||
      item.length ||
      item.smooth_factor ||
      item.meta?.continue_overlap_seconds != null
        ? [
            item.length != null ? `${item.length}f` : null,
            item.fps != null ? `${item.fps} fps` : null,
            item.smooth_factor != null ? `smooth ×${item.smooth_factor}` : null,
            item.meta?.continue_context_frames != null
              ? `ctx ${item.meta.continue_context_frames}f`
              : null,
            item.meta?.continue_overlap_seconds != null
              ? `blend ${item.meta.continue_overlap_seconds}s`
              : item.meta?.concat?.overlap_seconds != null
                ? `blend ${item.meta.concat.overlap_seconds}s`
                : null,
            item.meta?.concat?.mode ? String(item.meta.concat.mode) : null,
            item.meta?.context_windows?.enabled
              ? `ctx ${item.meta.context_windows.context_length}f∩${item.meta.context_windows.context_overlap}f`
              : null,
          ]
            .filter(Boolean)
            .join(" · ")
        : null,
    ],
    ["Parent", item.parent_job_id || item.meta?.parent_job_id || null],
    ["Prompt", item.prompt || item.positive || "—"],
    ["Negative", item.negative_prompt || item.negative || "—"],
    [
      "Positive (full)",
      item.positive && item.positive !== item.prompt ? item.positive : null,
    ],
    ["Job", item.id],
  ].filter((r) => r[1] != null);

  const dl = $("detail-meta");
  dl.innerHTML = "";
  for (const [k, v] of rows) {
    const dt = document.createElement("dt");
    dt.textContent = k;
    const dd = document.createElement("dd");
    dd.textContent = v;
    if (k === "Job" || k === "Seed" || k === "Checkpoint" || k === "Parent")
      dd.className = "mono";
    dl.appendChild(dt);
    dl.appendChild(dd);
  }

  renderUseIn(item);
}

function firstImageOf(item) {
  const o =
    (item.outputs || []).find((x) => x.kind === "image") ||
    (item.outputs || [])[0];
  if (!o?.url || isVideoOutput(o)) return null;
  return o;
}

/**
 * Build format-aware "Use in" actions for the selected gallery item.
 * Images → recipe / still modes; videos → Video→Video (+ prompt reuse).
 */
function renderUseIn(item) {
  const row = $("use-in-row");
  const hint = $("use-in-hint");
  if (!row) return;
  row.innerHTML = "";

  const hasImg = itemHasImage(item);
  const hasVid = itemHasVideo(item);
  const hasPrompt = !!(item.prompt || item.positive || item.meta?.positive);

  const actions = [];
  if (hasImg) {
    actions.push({
      id: "txt2img",
      label: "Text → Image",
      title: "Load preset/prompt/seed recipe",
    });
    actions.push({
      id: "img2vid",
      label: "Image → Video",
      title: "Use this still as I2V source (Draft)",
    });
  }
  if (hasVid) {
    actions.push({
      id: "vid2vid",
      label: "Video → Video",
      title: "Smooth, Continue, or Final on this clip",
    });
  }
  if (hasPrompt || hasVid || hasImg) {
    actions.push({
      id: "txt2vid",
      label: "Text → Video",
      title: hasPrompt
        ? "Reuse prompt/seed as a fresh T2V"
        : "Open Text → Video (fill prompt)",
    });
  }

  if (hint) {
    if (hasVid && hasImg) {
      hint.textContent = "Still + video — pick how to reuse.";
    } else if (hasVid) {
      hint.textContent = "Video — enhance via Video → Video, or reuse prompt.";
    } else if (hasImg) {
      hint.textContent = "Image — recipe reuse or animate with Image → Video.";
    } else {
      hint.textContent = "Limited media on this item.";
    }
  }

  if (!actions.length) {
    row.innerHTML = '<span class="meta">Nothing applicable for this item.</span>';
    return;
  }

  for (const a of actions) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "ghost";
    btn.textContent = a.label;
    btn.title = a.title;
    btn.onclick = () => useInMode(a.id);
    row.appendChild(btn);
  }
}

function useInMode(targetModality) {
  if (!selectedGallery) return;
  const item = selectedGallery;

  if (targetModality === "img2vid" && !itemHasImage(item)) {
    setStatus("No still on this item for Image → Video.", true);
    return;
  }
  if (targetModality === "vid2vid" && !itemHasVideo(item)) {
    setStatus("No video on this item for Video → Video.", true);
    return;
  }

  modality = targetModality;

  if (item.prompt) $("prompt").value = item.prompt;
  else if (item.positive) $("prompt").value = item.positive;
  if (item.negative_prompt) $("negative").value = item.negative_prompt;
  else if (item.negative) $("negative").value = item.negative;
  if (item.seed != null && item.seed !== "") $("seed").value = String(item.seed);

  if (targetModality === "txt2img") {
    if (item.width) $("width").value = String(item.width);
    if (item.height) $("height").value = String(item.height);
    if (item.preset && presets.some((p) => p.id === item.preset)) {
      selected = item.preset;
      const p = currentPreset();
      const fam = p?.base_family || "";
      const fx = Array.isArray(item.effects)
        ? item.effects.map((e) => (typeof e === "string" ? e : e.id)).filter(Boolean)
        : [];
      selectedEffects = new Set(
        fx.filter((id) => {
          const e = allEffects.find((x) => x.id === id);
          return e && effectCompatible(e, fam);
        })
      );
      renderPresets();
      renderEffects();
    }
    setStatus(`Loaded into Text → Image from ${item.id.slice(0, 8)}…`);
  } else if (targetModality === "txt2vid") {
    if (item.width) $("width").value = String(item.width);
    if (item.height) $("height").value = String(item.height);
    const prof = item.profile || item.meta?.profile;
    if (prof) videoProfile = prof;
    else videoProfile = "draft";
    setStatus(`Loaded prompt/seed into Text → Video · ${item.id.slice(0, 8)}…`);
  } else if (targetModality === "img2vid") {
    const imgOut = firstImageOf(item);
    if (!imgOut) {
      setStatus("No image on this item for Image → Video.", true);
      return;
    }
    setSourceFromGallery(item, imgOut);
    videoProfile = "draft";
    setStatus(
      `Source still set for Image → Video (Draft) · ${item.id.slice(0, 8)}…`
    );
  } else if (targetModality === "vid2vid") {
    if (!setSourceVideoFromGallery(item)) return;
    // Default op: smooth for quick enhance; user can switch Continue / Final
    if (!vid2vidOp) vid2vidOp = "smooth";
    setStatus(
      `Source clip set for Video → Video · ${vid2vidOp} · ${item.id.slice(0, 8)}…`
    );
  }

  renderModes();
  updateSizeHint();
  setPage("create");
  $("btn-gen").scrollIntoView({ behavior: "smooth", block: "center" });
}

async function loadPresets() {
  const data = await api("/v1/presets");
  presets = data.presets || [];
  renderPresets();
  renderEffects();
}

async function loadEffects() {
  const data = await api("/v1/effects");
  allEffects = data.effects || [];
  renderEffects();
}

async function loadCapabilities() {
  try {
    capabilities = await api("/v1/capabilities");
  } catch {
    capabilities = {
      modalities: MODES.map((m) => ({
        id: m.id,
        implemented: true,
        needs_preset: m.id === "txt2img",
        needs_source_image: m.id === "img2vid",
        needs_source_video: m.id === "vid2vid",
      })),
      video_packs: [{ id: "sample", defaults: { width: 640, height: 640 } }],
    };
  }
  // Ensure UI mode exists even if API still lists smooth/continue separately
  const mods = capabilities.modalities || [];
  if (!mods.some((m) => m.id === "vid2vid")) {
    mods.push({
      id: "vid2vid",
      label: "Video → Video",
      implemented: true,
      needs_source_video: true,
      ops: ["smooth", "continue", "finalize"],
      note: "Smooth / Continue / Final on a History clip",
    });
    capabilities.modalities = mods;
  }
  renderModes();
  // Seed clip-length field from pack default once
  const pack = videoPackDefaults();
  if ($("video-secs") && !$("video-secs").dataset.seeded) {
    $("video-secs").value = String(
      Math.round(wanLengthToSeconds(pack.length, pack.fps) * 10) / 10
    );
    $("video-secs").dataset.seeded = "1";
    updateVideoSecsHint();
  }
}

function gallerySentinelVisible() {
  const root = $("gallery-scroll");
  const sentinel = $("gallery-sentinel");
  if (!root || !sentinel) return false;
  const rr = root.getBoundingClientRect();
  const sr = sentinel.getBoundingClientRect();
  // Expand root box slightly (matches rootMargin intent)
  const pad = 140;
  return sr.top < rr.bottom + pad && sr.bottom > rr.top - pad;
}

async function loadGallery({ reset = false } = {}) {
  if (galleryLoading) return;
  if (!reset && !galleryHasMore) return;
  galleryLoading = true;
  const loadingEl = $("gallery-loading");
  if (loadingEl) loadingEl.hidden = false;
  try {
    if (reset) {
      galleryOffset = 0;
      galleryItems = [];
      galleryHasMore = true;
      galleryRenderedCount = 0;
    }
    const data = await api(
      `/v1/gallery?limit=${GALLERY_PAGE}&offset=${galleryOffset}&has_media=true`
    );
    const batch = data.items || [];
    galleryTotal = data.total || 0;
    if (reset) {
      galleryItems = batch;
      renderGallery({ reset: true });
    } else {
      // De-dupe by id in case of concurrent refresh races
      const seen = new Set(galleryItems.map((g) => g.id));
      for (const it of batch) {
        if (it?.id && !seen.has(it.id)) {
          galleryItems.push(it);
          seen.add(it.id);
        }
      }
      renderGallery({ reset: false });
    }
    galleryOffset = galleryItems.length;
    galleryHasMore = galleryOffset < galleryTotal && batch.length > 0;
  } catch (err) {
    if (reset) {
      // leave empty grid on hard fail
      renderGallery({ reset: true });
    }
    throw err;
  } finally {
    galleryLoading = false;
    if (loadingEl) loadingEl.hidden = true;
    // Sentinel often stays visible after a page loads — chain the next page
    // without waiting for another intersection event (fast scroll path).
    if (galleryHasMore && gallerySentinelVisible()) {
      queueMicrotask(() => {
        loadGallery({ reset: false }).catch((err) =>
          setStatus(String(err.message || err), true)
        );
      });
    }
  }
}

function setupGalleryInfiniteScroll() {
  const root = $("gallery-scroll");
  const sentinel = $("gallery-sentinel");
  if (!root || !sentinel) return;
  if (galleryObserver) galleryObserver.disconnect();
  galleryObserver = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (e.isIntersecting && galleryHasMore && !galleryLoading) {
          loadGallery({ reset: false }).catch((err) =>
            setStatus(String(err.message || err), true)
          );
        }
      }
    },
    { root, rootMargin: "140px", threshold: 0 }
  );
  galleryObserver.observe(sentinel);
}

function showResultMedia(url, kind) {
  const box = $("result");
  box.className = "result" + (isVideoUrl(url, kind) ? " has-video" : "");
  box.innerHTML = "";
  const bust = url + (url.includes("?") ? "&" : "?") + "t=" + Date.now();
  const el = mediaEl(bust, kind, {
    controls: true,
    autoplay: true,
    loop: true,
    muted: isVideoUrl(url, kind),
    preload: "auto",
    alt: "output",
  });
  if (el.tagName === "VIDEO") {
    el.controls = true;
    el.style.maxHeight = "min(70vh, 720px)";
    el.style.width = "100%";
  }
  box.appendChild(el);
}

function buildFinalizeBody() {
  if (!sourceVideo) throw new Error("Pick a source video first.");
  const m = sourceVideo.meta || {};
  const prompt =
    $("prompt").value.trim() || sourceVideo.prompt || m.positive || "";
  if (!prompt) throw new Error("Prompt required for Final.");

  const body = {
    prompt,
    seed: Number($("seed").value || -1),
    video_pack: "sample",
    video_profile: "final",
  };
  const neg = $("negative").value.trim() || sourceVideo.negative || "";
  if (neg) body.negative_prompt = neg;
  const w = $("width").value;
  const h = $("height").value;
  if (w) body.width = Number(w);
  else if (sourceVideo.width) body.width = Number(sourceVideo.width);
  if (h) body.height = Number(h);
  else if (sourceVideo.height) body.height = Number(sourceVideo.height);

  const srcMod = sourceVideo.srcModality || m.modality || "";
  const srcId = m.source_job_id || null;
  const srcInput = m.source_input || null;
  const srcName = m.source_filename || null;

  // Prefer original I2V path when the clip had a start frame
  if (srcMod === "img2vid" || srcId || srcInput) {
    body.modality = "img2vid";
    if (srcInput) body.source_input = srcInput;
    else if (srcId) {
      body.source_job_id = srcId;
      if (srcName) body.source_filename = srcName;
    } else {
      throw new Error(
        "This clip has no linked start image — use Text → Video Final or Continue instead."
      );
    }
  } else {
    body.modality = "txt2vid";
  }
  return body;
}

/** WAN frame length is 4n+1; map seconds → frames at pack fps. */
function secondsToWanLength(secs, fps) {
  const f = Number(fps) > 0 ? Number(fps) : 16;
  const n = Math.max(1, Math.round(Number(secs) * f));
  return Math.max(5, Math.floor((n - 1) / 4) * 4 + 1);
}

function wanLengthToSeconds(length, fps) {
  const f = Number(fps) > 0 ? Number(fps) : 16;
  return (Number(length) || 81) / f;
}

function updateVideoSecsHint() {
  const el = $("video-secs-hint");
  if (!el) return;
  const pack = videoPackDefaults();
  const secs = Number($("video-secs")?.value || wanLengthToSeconds(pack.length, pack.fps));
  const len = secondsToWanLength(secs, pack.fps);
  const win = Number($("context-length")?.value || 81);
  const willCtx =
    contextMode === "on" || (contextMode === "auto" && len > win);
  el.textContent =
    `→ ${len} frames @ ${pack.fps} fps (~${(len / pack.fps).toFixed(2)}s)` +
    (willCtx
      ? ` · context windows ON (window ${win}f)`
      : ` · single window` ) +
    ` · longer = more continuous (VRAM/time)`;
}

function renderContextModeChips() {
  const box = $("context-mode-chips");
  if (!box) return;
  box.innerHTML = "";
  for (const m of [
    { id: "auto", label: "Auto", title: "On when clip frames > window size" },
    { id: "on", label: "On", title: "Always use sliding context windows" },
    { id: "off", label: "Off", title: "Full attention only (fails/OOM on very long clips)" },
  ]) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "chip" + (contextMode === m.id ? " on" : "");
    btn.textContent = m.label;
    btn.title = m.title;
    btn.onclick = () => {
      contextMode = m.id;
      renderContextModeChips();
      updateVideoSecsHint();
      updateContextWindowsHint();
    };
    box.appendChild(btn);
  }
}

function renderContextOverlapPresets() {
  const box = $("context-overlap-presets");
  if (!box) return;
  box.innerHTML = "";
  const cur = Number($("context-overlap")?.value ?? 30);
  const win = Number($("context-length")?.value || 81);
  const presets = [
    { v: 16, label: "16f light" },
    { v: 30, label: "30f default" },
    { v: 40, label: "40f strong" },
    { v: Math.min(60, Math.max(8, Math.floor(win / 2))), label: "½ window" },
  ];
  for (const p of presets) {
    const btn = document.createElement("button");
    btn.type = "button";
    const on = Math.abs(cur - p.v) < 0.5;
    btn.className = "chip" + (on ? " on" : "");
    btn.textContent = p.label;
    btn.title = `${p.v} frames shared between windows (~${(p.v / (videoPackDefaults().fps || 16)).toFixed(1)}s)`;
    btn.onclick = () => {
      if ($("context-overlap")) $("context-overlap").value = String(p.v);
      renderContextOverlapPresets();
      updateContextWindowsHint();
    };
    box.appendChild(btn);
  }
}

function updateContextWindowsHint() {
  const el = $("context-windows-hint");
  if (!el) return;
  const pack = videoPackDefaults();
  const secs = Number($("video-secs")?.value || 5);
  const len = secondsToWanLength(secs, pack.fps);
  const win = Number($("context-length")?.value || 81);
  const ov = Number($("context-overlap")?.value || 30);
  const will =
    contextMode === "on" || (contextMode === "auto" && len > win);
  if (!will) {
    el.textContent =
      contextMode === "off"
        ? "Context windows off — full temporal attention (best quality under ~5s)."
        : `Clip fits in one window (${len}f ≤ ${win}f) — no sliding needed yet. Raise length to engage Auto.`;
    return;
  }
  const step = Math.max(1, win - ov);
  const nWin = Math.max(1, Math.ceil((len - ov) / step));
  el.textContent =
    `Active · ~${nWin} windows of ${win}f with ${ov}f overlap ` +
    `(~${(ov / pack.fps).toFixed(1)}s shared). Higher overlap = smoother long-clip continuity.`;
}

function renderContinueMethodChips() {
  const box = $("continue-method-chips");
  if (!box) return;
  box.innerHTML = "";
  for (const m of [
    {
      id: "flf_bridge",
      label: "FLF bridge",
      title: "Generate a short first–last-frame morph from parent tail to chunk head, then join",
    },
    {
      id: "prefix",
      label: "Prefix only",
      title: "Multi-frame I2V + xfade only (previous method)",
    },
  ]) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "chip" + (continueMethod === m.id ? " on" : "");
    btn.textContent = m.label;
    btn.title = m.title;
    btn.onclick = () => {
      continueMethod = m.id;
      renderContinueMethodChips();
      applyModeUI();
    };
    box.appendChild(btn);
  }
  const hint = $("continue-method-hint");
  if (hint) {
    hint.textContent =
      continueMethod === "flf_bridge"
        ? "Default · extra FLF pass (~1.5s) bridges real pixels of parent → new chunk. Slower, usually smoother."
        : "Faster · skip FLF; only motion-prefix I2V + seam blend.";
  }
}

function renderContextFramePresets() {
  const box = $("continue-context-presets");
  if (!box) return;
  box.innerHTML = "";
  const cur = Number($("continue-context")?.value ?? 16);
  const fps = videoPackDefaults().fps || 16;
  const presets = [
    { v: 1, label: "1f still" },
    { v: 12, label: "12f" },
    { v: 16, label: "16f" },
    { v: 24, label: "24f" },
    { v: 32, label: "32f" },
  ];
  for (const p of presets) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "chip" + (Math.abs(cur - p.v) < 0.5 ? " on" : "");
    btn.textContent = p.label;
    btn.title =
      p.v <= 1
        ? "Single last frame only — old behavior, weaker motion match"
        : `Feed last ${p.v} frames (~${(p.v / fps).toFixed(1)}s) as I2V motion prefix`;
    btn.onclick = () => {
      if ($("continue-context")) $("continue-context").value = String(p.v);
      renderContextFramePresets();
      updateContinueContextHint();
      updateSizeHint();
    };
    box.appendChild(btn);
  }
  updateContinueContextHint();
}

function updateContinueContextHint() {
  const hint = $("continue-context-hint");
  if (!hint) return;
  const n = Number($("continue-context")?.value ?? 16);
  const fps = videoPackDefaults().fps || 16;
  if (n <= 1) {
    hint.textContent = "Still-only continue — model restarts motion from a freeze (most jarring).";
    return;
  }
  hint.textContent =
    `Last ${n} frames (~${(n / fps).toFixed(1)}s) condition the chunk; prefix + ~8 warmup frames are stripped, then a real crossfade joins. ` +
    `If still jumpy: try 24–32f context and ~1s seam blend.`;
}

function renderOverlapPresets() {
  const box = $("continue-overlap-presets");
  if (!box) return;
  box.innerHTML = "";
  const cur = Number($("continue-overlap")?.value ?? 0.75);
  const presets = [
    { v: 0, label: "Hard cut" },
    { v: 0.5, label: "0.5s" },
    { v: 0.75, label: "0.75s" },
    { v: 1.0, label: "1s" },
    { v: 1.5, label: "1.5s" },
  ];
  for (const p of presets) {
    const btn = document.createElement("button");
    btn.type = "button";
    const on = Math.abs(cur - p.v) < 0.001;
    btn.className = "chip" + (on ? " on" : "");
    btn.textContent = p.label;
    btn.title =
      p.v <= 0
        ? "No pixel blend — abrupt join"
        : `Crossfade ~${p.v}s after motion-prefix continue (capped at 40% of either clip)`;
    btn.onclick = () => {
      if ($("continue-overlap")) $("continue-overlap").value = String(p.v);
      renderOverlapPresets();
      updateSizeHint();
    };
    box.appendChild(btn);
  }
  const hint = $("continue-overlap-hint");
  if (hint) {
    hint.textContent =
      "Pixel crossfade on the join only — motion continuity comes from Motion context above.";
  }
}

function applyVideoLengthToBody(body) {
  const pack = videoPackDefaults();
  const raw = $("video-secs")?.value;
  if (raw === undefined || raw === null || raw === "") return;
  const secs = Number(raw);
  if (!(secs > 0)) return;
  body.length = secondsToWanLength(secs, pack.fps);
  body.fps = pack.fps;
}

function applyContextWindowsToBody(body) {
  if (contextMode === "on") body.context_windows = true;
  else if (contextMode === "off") body.context_windows = false;
  // auto → omit (null) so server decides
  const cl = Number($("context-length")?.value);
  const co = Number($("context-overlap")?.value);
  if (cl > 0) body.context_length = Math.round(cl);
  if (!Number.isNaN(co) && co >= 0) body.context_overlap = Math.round(co);
}

function buildGenerateBody() {
  if (isVid2Vid()) {
    if (!sourceVideo?.jobId) {
      throw new Error("Pick a source video (History → Use in → Video → Video).");
    }
    if (vid2vidOp === "smooth") {
      return {
        modality: "smooth",
        parent_job_id: sourceVideo.jobId,
        smooth_factor: smoothFactor,
        prompt: "",
      };
    }
    if (vid2vidOp === "continue") {
      const secs = Number($("continue-secs")?.value || 5);
      let overlap = Number($("continue-overlap")?.value);
      if (Number.isNaN(overlap)) overlap = 0.75;
      let ctxF = Number($("continue-context")?.value);
      if (Number.isNaN(ctxF) || ctxF < 1) ctxF = 16;
      const body = {
        modality: "continue",
        parent_job_id: sourceVideo.jobId,
        continue_seconds: secs > 0 ? secs : 5,
        continue_overlap_seconds: Math.max(0, overlap),
        continue_context_frames: Math.round(ctxF),
        continue_method: continueMethod || "flf_bridge",
        continue_bridge_frames: 25,
        video_pack: "sample",
        video_profile: videoProfile || "draft",
        prompt: $("prompt").value.trim(),
        seed: Number($("seed").value || -1),
        effects: [...selectedEffects],
      };
      const neg = $("negative").value.trim();
      if (neg) body.negative_prompt = neg;
      const w = $("width").value;
      const h = $("height").value;
      if (w) body.width = Number(w);
      if (h) body.height = Number(h);
      return body;
    }
    if (vid2vidOp === "finalize") {
      const body = buildFinalizeBody();
      body.effects = [...selectedEffects];
      applyVideoLengthToBody(body);
      applyContextWindowsToBody(body);
      return body;
    }
    throw new Error(`Unknown Video → Video op: ${vid2vidOp}`);
  }

  const body = {
    modality,
    prompt: $("prompt").value.trim(),
    seed: Number($("seed").value || -1),
  };

  if (modality === "txt2img") {
    if (!selected) throw new Error("Pick a preset first.");
    if (!body.prompt) throw new Error("Enter a prompt.");
    body.preset = selected;
    body.effects = [...selectedEffects];
  } else if (modality === "txt2vid") {
    if (!body.prompt) throw new Error("Enter a video prompt.");
    body.video_pack = "sample";
    body.video_profile = videoProfile;
    body.effects = [...selectedEffects];
    applyVideoLengthToBody(body);
    applyContextWindowsToBody(body);
  } else if (modality === "img2vid") {
    if (!sourceImage) throw new Error("Pick a source from the gallery or disk.");
    if (sourceImage.type === "upload") body.source_input = sourceImage.input;
    else {
      body.source_job_id = sourceImage.jobId;
      body.source_filename = sourceImage.filename;
    }
    body.video_pack = "sample";
    body.video_profile = videoProfile;
    body.effects = [...selectedEffects];
    applyVideoLengthToBody(body);
    applyContextWindowsToBody(body);
  } else {
    throw new Error(`Unsupported mode: ${modality}`);
  }

  const neg = $("negative").value.trim();
  if (neg) body.negative_prompt = neg;
  const w = $("width").value;
  const h = $("height").value;
  if (w) body.width = Number(w);
  if (h) body.height = Number(h);
  return body;
}

function readBatchSize() {
  const el = $("batch-size");
  let n = Number(el?.value || 1);
  if (!Number.isFinite(n) || n < 1) n = 1;
  n = Math.min(8, Math.max(1, Math.round(n)));
  if (el) el.value = String(n);
  return n;
}

function describeEnqueue(jobs, snap) {
  const n = jobs.length || 1;
  const first = jobs[0];
  const short = first?.id ? String(first.id).slice(0, 8) : "";
  const activeId = snap?.active_job_id || snap?.active_job?.id;
  const startedNow =
    !!activeId && jobs.some((j) => j && j.id === activeId);
  const paused = !!snap?.paused;

  if (n > 1) {
    if (startedNow) {
      return `Added ${n} jobs to the queue · generating first now (${short})`;
    }
    if (paused) {
      return `Added ${n} jobs to the queue · paused at breakpoint`;
    }
    return `Added ${n} jobs to the queue · waiting (${short}…)`;
  }
  if (startedNow) {
    return `Added to queue · generating now (${short})`;
  }
  if (paused) {
    return `Added to queue · paused at breakpoint (${short})`;
  }
  return `Added to queue · waiting (${short})`;
}

async function generate() {
  let body;
  try {
    body = buildGenerateBody();
  } catch (e) {
    setStatus(String(e.message || e), true);
    return;
  }

  const batchSize = readBatchSize();
  body.batch_size = batchSize;

  $("btn-gen").disabled = true;
  try {
    const res = await api("/v1/generate", {
      method: "POST",
      body: JSON.stringify(body),
    });
    const jobs = res.jobs || (res.id ? [res] : []);
    for (const j of jobs) {
      if (j?.id) sessionJobIds.add(j.id);
    }
    // Brief beat so the worker can claim the first job when idle
    await new Promise((r) => setTimeout(r, 120));
    await refreshQueue();
    const snap = lastQueueSnap;
    setStatus(describeEnqueue(jobs, snap));
    if ($("result")?.classList.contains("empty")) {
      $("result").textContent = "In queue — see the Queue panel.";
    }
  } catch (e) {
    setStatus(String(e.message || e), true);
  } finally {
    $("btn-gen").disabled = false;
  }
}

// Wire UI
$("btn-gen").onclick = generate;
if ($("tab-create")) {
  $("tab-create").onclick = () => setPage("create");
}
if ($("tab-history")) {
  $("tab-history").onclick = () => setPage("history");
}
if ($("btn-cancel-active")) {
  $("btn-cancel-active").onclick = () => {
    const id = $("btn-cancel-active").dataset.jobId;
    cancelJobById(id);
  };
}
if ($("btn-add-break")) {
  $("btn-add-break").onclick = async () => {
    try {
      await api("/v1/queue/breakpoint", {
        method: "POST",
        body: JSON.stringify({ label: "Break" }),
      });
      await refreshQueue();
      setStatus("Breakpoint added to queue.");
    } catch (e) {
      setStatus(String(e.message || e), true);
    }
  };
}
if ($("btn-unload")) {
  $("btn-unload").onclick = async () => {
    try {
      await api("/v1/admin/unload-models", { method: "POST" });
      setStatus("Models unloaded from VRAM.");
      await refreshQueue();
    } catch (e) {
      setStatus(String(e.message || e), true);
    }
  };
}
$("btn-refresh-gallery").onclick = () =>
  loadGallery({ reset: true }).catch((e) => setStatus(String(e.message || e), true));
$("btn-close-detail").onclick = () => {
  $("detail-panel").hidden = true;
  const ph = $("detail-placeholder");
  if (ph) ph.hidden = false;
  selectedGallery = null;
  updateGallerySelection();
};
$("btn-pick-file").onclick = () => $("file-source").click();
$("file-source").onchange = async (ev) => {
  const f = ev.target.files?.[0];
  ev.target.value = "";
  if (!f) return;
  try {
    if (modality !== "img2vid") {
      modality = "img2vid";
      renderModes();
    }
    await uploadSourceFile(f);
  } catch (e) {
    setStatus(String(e.message || e), true);
  }
};
$("btn-reset-width").onclick = (e) => {
  e.stopPropagation();
  toggleResetMenu("width");
};
$("btn-reset-height").onclick = (e) => {
  e.stopPropagation();
  toggleResetMenu("height");
};
document.addEventListener("click", (e) => {
  if (!e.target.closest(".reset-wrap")) closeResetMenus();
});
$("width").addEventListener("input", updateSizeHint);
$("height").addEventListener("input", updateSizeHint);
if ($("continue-secs")) {
  $("continue-secs").addEventListener("input", updateSizeHint);
}
if ($("continue-context")) {
  $("continue-context").addEventListener("input", () => {
    renderContextFramePresets();
    updateSizeHint();
  });
}
if ($("continue-overlap")) {
  $("continue-overlap").addEventListener("input", () => {
    renderOverlapPresets();
    updateSizeHint();
  });
}
if ($("video-secs")) {
  $("video-secs").addEventListener("input", () => {
    updateVideoSecsHint();
    updateContextWindowsHint();
  });
}
if ($("context-length")) {
  $("context-length").addEventListener("input", () => {
    renderContextOverlapPresets();
    updateVideoSecsHint();
    updateContextWindowsHint();
  });
}
if ($("context-overlap")) {
  $("context-overlap").addEventListener("input", () => {
    renderContextOverlapPresets();
    updateContextWindowsHint();
  });
}

$("btn-copy-prompt").onclick = async () => {
  if (!selectedGallery) return;
  const t = selectedGallery.prompt || selectedGallery.positive || "";
  try {
    await navigator.clipboard.writeText(t);
    setStatus("Prompt copied.");
  } catch {
    setStatus("Could not copy.", true);
  }
};
$("btn-copy-seed").onclick = async () => {
  if (!selectedGallery || selectedGallery.seed == null) return;
  try {
    await navigator.clipboard.writeText(String(selectedGallery.seed));
    setStatus("Seed copied.");
  } catch {
    setStatus("Could not copy.", true);
  }
};

// Restore page from hash without wiping form/queue state
if (location.hash === "#history") setPage("history");
else setPage("create");
window.addEventListener("hashchange", () => {
  if (location.hash === "#history") setPage("history");
  else if (location.hash === "#create" || location.hash === "") setPage("create");
});

refreshHealth();
setInterval(refreshHealth, 15000);
startQueuePolling();
refreshQueue().catch(() => {});

Promise.all([
  loadCapabilities(),
  loadPresets(),
  loadEffects(),
  loadGallery({ reset: true }),
])
  .then(() => setupGalleryInfiniteScroll())
  .catch((e) => setStatus(String(e.message || e), true));
