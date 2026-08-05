/*
 * app.js
 * =======
 * Logica del frontend "Behaviour Vision Lab": legge lo stato dei controlli
 * della sidebar, chiama il bridge Python (webui/api.py, esposto come
 * `window.pywebview.api.<metodo>(...)`, ognuno una promise), e aggiorna il
 * DOM (frame video, timeline, status bar) in risposta ai payload ricevuti.
 *
 * Due percorsi diversi per far arrivare un frame sullo schermo:
 *  1. Play: `api.play(fps)` avvia un thread Python in background che spinge
 *     ogni frame da solo chiamando `window.onPipelineFrame(payload)` (vedi
 *     webui/api.py::Api._push_frame) -- qui non c'e' nessun polling lato JS.
 *  2. Back/Forward/Seek: la chiamata stessa (`api.step_forward()` ecc.)
 *     RESTITUISCE il payload del frame, quindi lo si passa direttamente a
 *     `onPipelineFrame` invece di aspettare una spinta separata.
 * In entrambi i casi il payload ha la stessa forma {ok, frame, status}, cosi'
 * il rendering ha un solo punto d'ingresso.
 */

(() => {
  "use strict";

  const state = {
    playing: false,
    hasPlayer: false,
    lastKnownCursor: 0,
    cachedFrameCount: 0,
  };

  // ---------------------------------------------------------------- utils
  function $(id) { return document.getElementById(id); }

  function api() {
    // window.pywebview.api non e' garantito pronto finche' l'evento
    // 'pywebviewready' non e' scattato (vedi bindReady sotto) -- ma anche
    // dopo, se il modulo viene aperto in un browser normale per debug (senza
    // pywebview) l'oggetto semplicemente non esiste: qui lo segnaliamo
    // chiaramente invece di far fallire silenziosamente ogni bottone.
    if (!window.pywebview || !window.pywebview.api) {
      throw new Error("pywebview bridge not available (opened outside the app window?)");
    }
    return window.pywebview.api;
  }

  function setStatusPill(text, kind) {
    const pill = $("pill-status");
    pill.textContent = text;
    pill.className = "pill " + (kind || "pill-dim");
  }

  function formatTimecode(seconds) {
    const s = Math.max(0, seconds || 0);
    const m = Math.floor(s / 60);
    const rem = (s - m * 60).toFixed(1);
    return String(m).padStart(2, "0") + ":" + String(rem).padStart(4, "0");
  }

  // ------------------------------------------------------- feature gating
  // Stessa regola di gui/app.py::_on_mode_change: mani/viso valgono solo in
  // Pose estimation / Both, MediaPipe pose-per-maschera solo in Segmentation.
  function applyModeGating() {
    const mode = $("mode-select").value;
    const poseCapable = mode === "pose" || mode === "both";
    const segOnly = mode === "segmentation";

    const featureToggles = [
      "hands-toggle", "eyes-toggle", "mouth-toggle", "eyebrows-toggle", "head-movement-toggle",
    ];
    featureToggles.forEach((id) => {
      const el = $(id);
      el.disabled = !poseCapable;
      if (!poseCapable) el.checked = false;
    });
    $("features-card").classList.toggle("disabled", !poseCapable);

    $("mediapipe-pose-toggle").disabled = !segOnly;
    if (!segOnly) $("mediapipe-pose-toggle").checked = false;
    $("seg-extras-card").classList.toggle("disabled", !segOnly);

    $("pill-mode").textContent =
      mode === "pose" ? "Pose estimation" : mode === "both" ? "Both" : "Segmentation";

    updatePipelineFlow();
  }

  function updatePipelineFlow() {
    // Diagramma di flusso REALE, costruito dalla configurazione corrente
    // della sidebar -- non un testo fisso "YOLO26 Segment -> Tracking ->
    // MediaPipe" come nel mock, che rifletterebbe passi non davvero attivi
    // se l'utente scegliesse un'altra combinazione.
    const mode = $("mode-select").value;
    const scale = $("scale-select").value;
    const steps = [];

    if (mode === "segmentation" || mode === "both") {
      steps.push(`YOLO26${scale} Segment`);
    }
    if (mode === "pose" || mode === "both") {
      steps.push(`YOLO26${scale} Pose`);
    }
    steps.push("ByteTrack");
    if ($("reid-toggle").checked && $("max-people-input").value) {
      steps.push("Re-ID");
    }
    if (mode === "segmentation" && $("mediapipe-pose-toggle").checked) {
      steps.push("MediaPipe Pose");
    }
    if (poseFeaturesActive()) {
      steps.push("MediaPipe Face/Hands");
    }

    const flow = $("pipeline-flow");
    flow.innerHTML = "";
    steps.forEach((step, i) => {
      if (i > 0) {
        const arrow = document.createElement("span");
        arrow.className = "flow-arrow";
        arrow.textContent = "→";
        flow.appendChild(arrow);
      }
      const el = document.createElement("span");
      el.className = "flow-step";
      el.textContent = step;
      flow.appendChild(el);
    });
  }

  function poseFeaturesActive() {
    return ["hands-toggle", "eyes-toggle", "mouth-toggle", "eyebrows-toggle", "head-movement-toggle"]
      .some((id) => $(id).checked && !$(id).disabled);
  }

  // ------------------------------------------------------------- collect
  function collectParams() {
    return {
      mode: $("mode-select").value,
      fps: $("fps-input").value,
      device: "mps",
      scale: $("scale-select").value,
      max_people: $("max-people-input").value,
      reid: $("reid-toggle").checked,
      with_hands: $("hands-toggle").checked,
      with_eyes: $("eyes-toggle").checked,
      with_mouth: $("mouth-toggle").checked,
      with_eyebrows: $("eyebrows-toggle").checked,
      with_head_movement: $("head-movement-toggle").checked,
      with_mediapipe_pose: $("mediapipe-pose-toggle").checked,
    };
  }

  // ---------------------------------------------------------------- render
  window.onPipelineFrame = function onPipelineFrame(payload) {
    if (!payload) return;
    if (payload.frame) {
      const img = $("video-frame");
      img.src = payload.frame;
      img.classList.remove("hidden");
      $("video-placeholder").style.display = "none";
    }
    if (payload.status) {
      renderStatus(payload.status);
    }
    if (payload.ok === false && payload.error) {
      setStatusPill(payload.error, "pill-dim");
    }
  };

  function renderStatus(status) {
    if (typeof status.frame_index === "number") {
      $("metric-frame").textContent = status.frame_index >= 0 ? status.frame_index : "–";
      state.lastKnownCursor = Math.max(0, status.frame_index);
      state.cachedFrameCount = Math.max(state.cachedFrameCount, status.frame_index + 1);
      updateTimeline();
    }
    if (typeof status.timecode_s === "number") {
      $("timecode").textContent = formatTimecode(status.timecode_s);
    }
    if (typeof status.processing_fps === "number") {
      $("metric-fps").textContent = status.processing_fps > 0 ? status.processing_fps.toFixed(1) : "–";
    }
    if (typeof status.avg_latency_ms === "number") {
      $("metric-latency").textContent = status.avg_latency_ms > 0 ? status.avg_latency_ms.toFixed(0) + " ms" : "–";
    }
    if (typeof status.people_count === "number") {
      $("metric-people").textContent = status.people_count;
    }
    if (status.device) {
      $("metric-device").textContent = status.device;
      $("pill-device").textContent = status.device;
    }
    if (status.is_finished) {
      state.playing = false;
      $("btn-play").textContent = "▶";
      setStatusPill("Finished", "pill-dim");
    } else if (state.playing) {
      setStatusPill("Playing", "pill-live");
    } else {
      setStatusPill("Paused", "pill-dim");
    }
  }

  function updateTimeline() {
    // Lo scrubber copre SOLO il prefisso gia' elaborato (vedi
    // gui/video_player.py::seek): non conosciamo la lunghezza totale del
    // video in anticipo (i tracker sono sequenziali/stateful, niente conteggio
    // frame pre-calcolato), quindi la barra rappresenta "quanto e' stato
    // elaborato finora", non "quanto manca alla fine".
    const progressPct = state.cachedFrameCount > 0
      ? (state.lastKnownCursor + 1) / state.cachedFrameCount * 100
      : 0;
    $("timeline-progress").style.width = progressPct + "%";
    $("timeline-cursor").style.left = progressPct + "%";
  }

  // ----------------------------------------------------------- transport
  async function onLoadVideo() {
    try {
      const path = await api().pick_video_file();
      if (!path) return;
      $("video-filename").textContent = path.split("/").pop();
      $("video-check").classList.remove("hidden");
      state.hasPlayer = false;
      setStatusPill("Video loaded", "pill-dim");
    } catch (err) {
      setStatusPill(String(err.message || err), "pill-dim");
    }
  }

  async function ensurePlayerBuilt() {
    if (state.hasPlayer) return true;
    return applySettings();
  }

  async function applySettings() {
    try {
      const result = await api().build_player(collectParams());
      if (!result.ok) {
        setStatusPill(result.error || "Could not build player", "pill-dim");
        return false;
      }
      state.hasPlayer = true;
      state.playing = false;
      state.lastKnownCursor = 0;
      state.cachedFrameCount = 0;
      $("btn-play").textContent = "▶";
      updateTimeline();
      setStatusPill("Ready", "pill-dim");
      return true;
    } catch (err) {
      setStatusPill(String(err.message || err), "pill-dim");
      return false;
    }
  }

  async function onPlayPause() {
    if (state.playing) {
      state.playing = false;
      $("btn-play").textContent = "▶";
      setStatusPill("Paused", "pill-dim");
      try { await api().pause(); } catch (err) { /* window closing, etc. */ }
      return;
    }
    const ok = await ensurePlayerBuilt();
    if (!ok) return;
    state.playing = true;
    $("btn-play").textContent = "⏸";
    setStatusPill("Playing", "pill-live");
    const speed = parseFloat($("speed-select").value) || 1;
    const sourceFps = parseFloat($("fps-input").value) || 15;
    try {
      await api().play(sourceFps * speed);
    } catch (err) {
      state.playing = false;
      $("btn-play").textContent = "▶";
      setStatusPill(String(err.message || err), "pill-dim");
    }
  }

  async function onStepForward() {
    const ok = await ensurePlayerBuilt();
    if (!ok) return;
    state.playing = false;
    $("btn-play").textContent = "▶";
    try {
      const payload = await api().step_forward();
      window.onPipelineFrame(payload);
    } catch (err) {
      setStatusPill(String(err.message || err), "pill-dim");
    }
  }

  async function onStepBack() {
    if (!state.hasPlayer) return;
    state.playing = false;
    $("btn-play").textContent = "▶";
    try {
      const payload = await api().step_back();
      window.onPipelineFrame(payload);
    } catch (err) {
      setStatusPill(String(err.message || err), "pill-dim");
    }
  }

  async function onTimelineClick(evt) {
    if (!state.hasPlayer || state.cachedFrameCount === 0) return;
    const track = $("timeline-track");
    const rect = track.getBoundingClientRect();
    const ratio = Math.min(1, Math.max(0, (evt.clientX - rect.left) / rect.width));
    const targetIndex = Math.round(ratio * (state.cachedFrameCount - 1));
    state.playing = false;
    $("btn-play").textContent = "▶";
    try {
      const payload = await api().seek(targetIndex);
      if (payload.ok === false) {
        // fuori dal prefisso gia' elaborato: elabora in sequenza fino li',
        // cosi' l'utente vede il recupero avanzare invece di un salto
        // istantaneo (vedi api.py::Api.seek).
        setStatusPill("Catching up…", "pill-dim");
        while (state.cachedFrameCount <= targetIndex) {
          const step = await api().step_forward();
          window.onPipelineFrame(step);
          if (step.status && step.status.is_finished) break;
        }
      } else {
        window.onPipelineFrame(payload);
      }
    } catch (err) {
      setStatusPill(String(err.message || err), "pill-dim");
    }
  }

  async function onExportCsv() {
    try {
      const path = await api().pick_save_csv_path();
      if (!path) return;
      const result = await api().export_csv(path);
      setStatusPill(result.ok ? `Saved ${result.rows} rows` : (result.error || "Save failed"), "pill-dim");
    } catch (err) {
      setStatusPill(String(err.message || err), "pill-dim");
    }
  }

  function onFullscreen() {
    const wrap = $("video-frame-wrap");
    if (!document.fullscreenElement) {
      wrap.requestFullscreen && wrap.requestFullscreen();
    } else {
      document.exitFullscreen && document.exitFullscreen();
    }
  }

  // -------------------------------------------------------------- wiring
  function wireEvents() {
    $("btn-load-video").addEventListener("click", onLoadVideo);
    $("btn-apply").addEventListener("click", applySettings);
    $("btn-play").addEventListener("click", onPlayPause);
    $("btn-forward").addEventListener("click", onStepForward);
    $("btn-back").addEventListener("click", onStepBack);
    $("btn-export-csv").addEventListener("click", onExportCsv);
    $("btn-fullscreen").addEventListener("click", onFullscreen);
    $("timeline-track").addEventListener("click", onTimelineClick);
    $("mode-select").addEventListener("change", applyModeGating);
    ["arch-select"].forEach((id) => {
      $(id).addEventListener("change", () => {
        // SAM3 gira sulle macchine GPU dedicate del gruppo di ricerca, non
        // qui -- stesso avviso di gui/app.py::_on_arch_change.
        if ($("arch-select").value === "sam3") {
          setStatusPill("SAM3 not available here — staying on YOLO", "pill-dim");
          $("arch-select").value = "yolo";
        }
      });
    });
    // qualunque cambio di parametro invalida il player corrente: bisogna
    // premere "Apply settings" per ricostruirlo (un tracker gia' avviato non
    // si puo' riconfigurare a meta' strada, vedi video_player.py) -- qui ci
    // limitiamo a segnalarlo, senza ricostruire da soli ad ogni click.
    const invalidatingIds = [
      "arch-select", "mode-select", "scale-select", "max-people-input",
      "reid-toggle", "hands-toggle", "eyes-toggle", "mouth-toggle",
      "eyebrows-toggle", "head-movement-toggle", "mediapipe-pose-toggle",
    ];
    invalidatingIds.forEach((id) => {
      $(id).addEventListener("change", () => {
        state.hasPlayer = false;
        updatePipelineFlow();
      });
    });
  }

  function init() {
    wireEvents();
    applyModeGating();
    updateTimeline();
  }

  if (window.pywebview) {
    init();
  } else {
    // in fase di sviluppo il file puo' essere aperto in un browser normale
    // per controllare rapidamente il layout: i controlli funzionano, le
    // chiamate all'API falliscono con un messaggio chiaro invece di un
    // errore silenzioso (vedi api()).
    document.addEventListener("DOMContentLoaded", init);
  }
  window.addEventListener("pywebviewready", init);
})();
