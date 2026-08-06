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
 *
 * Due bottoni Play, uno stato solo: "Play analysis" nella sidebar e il
 * pulsante circolare sotto il video controllano LO STESSO play/pause (vedi
 * onPlayPause) e restano sincronizzati -- nel mock sono due controlli
 * distinti ma non c'e' motivo che rappresentino stati diversi.
 */

(() => {
  "use strict";

  const state = {
    playing: false,
    hasPlayer: false,
    lastKnownCursor: 0,
    cachedFrameCount: 0,
    totalFrameCount: null,   // da probe_video_metadata via pick_video_file() -- solo informativo
    totalDurationS: null,
    lastTimecodeS: 0,
    maxPeople: null,
    detectedDevice: null,  // da Api.detect_device() (vedi init()) -- usato SOLO per
    // abilitare/disabilitare SAM 3.1/SAM2 nel selettore Architecture, il
    // controllo definitivo resta lato server in Api.build_player().
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
    // "kind" e' "live" (in riproduzione, pallino acceso) o "idle" (pallino
    // spento).
    $("status-label").textContent = text;
    $("dot-status").className = "status-dot " + (kind === "live" ? "status-dot-live" : "status-dot-idle");
  }

  function formatTimecode(seconds) {
    // "HH:MM:SS.cc" (centesimi), stesso formato del mock ("00:00:06.30").
    const s = Math.max(0, seconds || 0);
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const rem = s % 60;
    const cc = Math.round((rem - Math.floor(rem)) * 100);
    return [h, m, Math.floor(rem)].map((v) => String(v).padStart(2, "0")).join(":")
      + "." + String(cc).padStart(2, "0");
  }

  function formatTimeShort(seconds) {
    // "MM:SS" per le tacche della timeline (o "H:MM:SS" se serve).
    const s = Math.max(0, Math.round(seconds || 0));
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = s % 60;
    const mm = h > 0 ? String(m).padStart(2, "0") : String(m);
    const parts = h > 0 ? [String(h), mm, String(sec).padStart(2, "0")] : [mm, String(sec).padStart(2, "0")];
    return parts.join(":");
  }

  function setPlayIcon(playing) {
    const glyphPlay = '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>';
    const glyphPause = '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M7 5h4v14H7zM13 5h4v14h-4z"/></svg>';
    $("btn-play").innerHTML = playing ? glyphPause : glyphPlay;
    $("btn-play-analysis").innerHTML =
      (playing ? glyphPause : glyphPlay) + (playing ? " Pause" : " Play analysis");
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

    applyArchGating();
    updatePipelineFlow();
  }

  // Backend di segmentazione (YOLO/SAM 3.1/SAM2, vedi
  // segmentation/sam_backend.py): SAM 3.1/SAM2 valgono solo in
  // Segmentation/Both E richiedono una GPU CUDA -- qui si disabilitano le
  // opzioni non valide e si torna a "yolo" se quella selezionata smette di
  // esserlo (es. l'utente passa a modalita' Pose). Il controllo DEFINITIVO
  // resta comunque lato server in Api.build_player() (vedi il suo
  // docstring): qui e' solo per non mostrare in UI una scelta che
  // fallirebbe subito.
  function applyArchGating() {
    const archSelect = $("arch-select");
    const mode = $("mode-select").value;
    const segCapable = mode === "segmentation" || mode === "both";
    const cudaAvailable = state.detectedDevice === "cuda";

    ["sam31", "sam2"].forEach((value) => {
      const opt = archSelect.querySelector(`option[value="${value}"]`);
      if (opt) opt.disabled = !cudaAvailable;
    });

    const archIsSam = archSelect.value === "sam31" || archSelect.value === "sam2";
    if (archIsSam && !cudaAvailable) {
      setStatusPill(`${archSelect.options[archSelect.selectedIndex].text} needs a CUDA GPU — staying on YOLO`, "idle");
      archSelect.value = "yolo";
    } else if (archIsSam && !segCapable) {
      setStatusPill(`${archSelect.options[archSelect.selectedIndex].text} only applies to Segmentation/Both — staying on YOLO`, "idle");
      archSelect.value = "yolo";
    }

    const showChunkFields = archSelect.value === "sam31" || archSelect.value === "sam2";
    $("sam-chunk-fields").classList.toggle("hidden", !showChunkFields);
    $("arch-hint").classList.toggle("hidden", cudaAvailable);
  }

  const FLOW_ICONS = {
    box: '<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 2 3 7v10l9 5 9-5V7l-9-5Z" stroke="white" stroke-width="1.8" stroke-linejoin="round"/><path d="M3 7l9 5 9-5" stroke="white" stroke-width="1.8" stroke-linejoin="round"/></svg>',
    target: '<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="8" stroke="white" stroke-width="1.8"/><circle cx="12" cy="12" r="2.4" fill="white"/></svg>',
    link: '<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M9 15l6-6M8 16l-2 2a3.5 3.5 0 0 1-5-5l2-2M16 8l2-2a3.5 3.5 0 0 1 5 5l-2 2" stroke="white" stroke-width="1.8" stroke-linecap="round"/></svg>',
    nodes: '<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="5" cy="7" r="2" fill="white"/><circle cx="19" cy="7" r="2" fill="white"/><circle cx="12" cy="18" r="2" fill="white"/><path d="M6.6 8.2 10.5 16M17.4 8.2 13.5 16M7 7h10" stroke="white" stroke-width="1.4"/></svg>',
  };

  function updatePipelineFlow() {
    // Diagramma di flusso REALE, costruito dalla configurazione corrente
    // della sidebar -- non un testo fisso "YOLO26 Segment -> Tracking ->
    // MediaPipe" come nel mock, che rifletterebbe passi non davvero attivi
    // se l'utente scegliesse un'altra combinazione.
    const mode = $("mode-select").value;
    const scale = $("scale-select").value;
    const steps = [];

    if (mode === "segmentation" || mode === "both") {
      const arch = $("arch-select").value;
      const segLabel = arch === "sam31" ? "SAM 3.1 Segment"
        : arch === "sam2" ? "SAM2 Segment"
        : `YOLO26${scale} Segment`;
      steps.push({ label: segLabel, icon: "box", cls: "seg" });
    }
    if (mode === "pose" || mode === "both") {
      steps.push({ label: `YOLO26${scale} Pose`, icon: "box", cls: "pose" });
    }
    steps.push({ label: "ByteTrack", icon: "target", cls: "track" });
    if ($("reid-toggle").checked && $("max-people-input").value) {
      steps.push({ label: "Re-ID", icon: "link", cls: "reid" });
    }
    if (mode === "segmentation" && $("mediapipe-pose-toggle").checked) {
      steps.push({ label: "MediaPipe Pose", icon: "nodes", cls: "face" });
    }
    if (poseFeaturesActive()) {
      steps.push({ label: "MediaPipe Face/Hands", icon: "nodes", cls: "face" });
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
      const iconWrap = document.createElement("span");
      iconWrap.className = "flow-step-icon " + step.cls;
      iconWrap.innerHTML = FLOW_ICONS[step.icon];
      el.appendChild(iconWrap);
      el.appendChild(document.createTextNode(step.label));
      flow.appendChild(el);
    });
  }

  function poseFeaturesActive() {
    return ["hands-toggle", "eyes-toggle", "mouth-toggle", "eyebrows-toggle", "head-movement-toggle"]
      .some((id) => $(id).checked && !$(id).disabled);
  }

  // ------------------------------------------------------------- collect
  function collectParams() {
    // Nessun campo "device": lo lasciamo assente cosi' Api.build_player()
    // lo auto-rileva lato Python (cuda/mps/cpu, vedi common/device.py) --
    // prima era fisso a "mps" qui, il che rompeva silenziosamente su una
    // macchina con GPU CUDA e nessun Metal.
    return {
      mode: $("mode-select").value,
      fps: $("fps-input").value,
      scale: $("scale-select").value,
      max_people: $("max-people-input").value,
      reid: $("reid-toggle").checked,
      with_hands: $("hands-toggle").checked,
      with_eyes: $("eyes-toggle").checked,
      with_mouth: $("mouth-toggle").checked,
      with_eyebrows: $("eyebrows-toggle").checked,
      with_head_movement: $("head-movement-toggle").checked,
      with_mediapipe_pose: $("mediapipe-pose-toggle").checked,
      seg_backend: $("arch-select").value,
      sam_chunk_size: $("sam-chunk-size-input").value,
      sam_overlap: $("sam-overlap-input").value,
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
      setStatusPill(payload.error, "idle");
    }
  };

  function renderStatus(status) {
    if (typeof status.total_frame_count === "number") state.totalFrameCount = status.total_frame_count;
    if (typeof status.total_duration_s === "number") state.totalDurationS = status.total_duration_s;
    if (typeof status.max_people === "number") state.maxPeople = status.max_people;

    if (typeof status.frame_index === "number" && status.frame_index >= 0) {
      state.lastKnownCursor = status.frame_index;
      state.cachedFrameCount = Math.max(state.cachedFrameCount, status.frame_index + 1);
      $("metric-frame").textContent = state.totalFrameCount
        ? `${status.frame_index} / ${state.totalFrameCount}`
        : String(status.frame_index);
    }
    if (typeof status.timecode_s === "number") {
      state.lastTimecodeS = status.timecode_s;
      const totalLabel = state.totalDurationS != null ? formatTimecode(state.totalDurationS) : "--:--:--.--";
      $("timecode").textContent = `${formatTimecode(status.timecode_s)} / ${totalLabel}`;
    }
    updateTimeline();

    if (typeof status.processing_fps === "number") {
      $("metric-fps").textContent = status.processing_fps > 0 ? status.processing_fps.toFixed(1) + " FPS" : "–";
    }
    if (typeof status.avg_latency_ms === "number") {
      $("metric-latency").textContent = status.avg_latency_ms > 0 ? status.avg_latency_ms.toFixed(0) + " ms" : "–";
    }
    if (typeof status.people_count === "number") {
      $("metric-people").textContent = state.maxPeople ? `${status.people_count} / ${state.maxPeople}` : String(status.people_count);
      $("flow-tracks-label").textContent = `${status.people_count} active track${status.people_count === 1 ? "" : "s"}`;
      $("dot-tracks-active").className = "status-dot " + (status.people_count > 0 ? "status-dot-live" : "status-dot-idle");
    }
    if (status.device) {
      $("gpu-indicator").title = `Configured device: ${status.device}`;
    }
    if (status.is_finished) {
      state.playing = false;
      setPlayIcon(false);
      setStatusPill("Finished", "idle");
    } else if (state.playing) {
      setStatusPill("Playing", "live");
    } else {
      setStatusPill("Paused", "idle");
    }
  }

  function updateTimeline() {
    // Se la durata totale e' nota (letta dai metadati del file al momento
    // del caricamento, vedi api.py::probe_video_metadata) la barra mostra
    // "quanto del video intero e' stato elaborato finora"; altrimenti (rari
    // container senza questi metadati) ripiega su "quanto della cache
    // corrente e' stato visto", che e' comunque tutto cio' che si puo'
    // sapere. In ENTRAMBI i casi lo scrubber resta cliccabile SOLO nel
    // prefisso gia' elaborato -- vedi onTimelineClick.
    let progressPct;
    if (state.totalDurationS) {
      progressPct = Math.min(100, (state.lastTimecodeS / state.totalDurationS) * 100);
    } else {
      progressPct = state.cachedFrameCount > 0 ? (state.lastKnownCursor + 1) / state.cachedFrameCount * 100 : 0;
    }
    $("timeline-progress").style.width = progressPct + "%";
    $("timeline-cursor").style.left = progressPct + "%";
  }

  function buildTimelineTicks() {
    const ticksEl = $("timeline-ticks");
    ticksEl.innerHTML = "";
    const total = state.totalDurationS;
    if (!total) return;
    const tickCount = 5;
    for (let i = 0; i <= tickCount; i++) {
      const t = (total * i) / tickCount;
      const span = document.createElement("span");
      span.textContent = formatTimeShort(t);
      ticksEl.appendChild(span);
    }
  }

  // ----------------------------------------------------------- transport
  async function onLoadVideo() {
    try {
      const info = await api().pick_video_file();
      if (!info || !info.path) return;
      $("video-filename").textContent = info.path.split("/").pop();
      $("video-check").classList.remove("hidden");
      state.hasPlayer = false;
      state.totalFrameCount = info.frame_count || null;
      state.totalDurationS = info.duration_s || null;
      state.cachedFrameCount = 0;
      state.lastKnownCursor = 0;
      state.lastTimecodeS = 0;
      buildTimelineTicks();
      updateTimeline();
      $("timecode").textContent = `${formatTimecode(0)} / ${state.totalDurationS != null ? formatTimecode(state.totalDurationS) : "--:--:--.--"}`;
      $("metric-frame").textContent = state.totalFrameCount ? `0 / ${state.totalFrameCount}` : "–";
      setStatusPill("Video loaded", "idle");
    } catch (err) {
      setStatusPill(String(err.message || err), "idle");
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
        setStatusPill(result.error || "Could not build player", "idle");
        return false;
      }
      state.hasPlayer = true;
      state.playing = false;
      state.lastKnownCursor = 0;
      state.cachedFrameCount = 0;
      state.lastTimecodeS = 0;
      setPlayIcon(false);
      updateTimeline();
      setStatusPill("Ready", "idle");
      return true;
    } catch (err) {
      setStatusPill(String(err.message || err), "idle");
      return false;
    }
  }

  async function onRestart() {
    // "Restart pipeline": ricostruzione forzata da zero, anche se un player
    // esisteva gia' -- un tracker gia' avviato non si puo' riconfigurare a
    // meta' strada (vedi gui/video_player.py), stesso ruolo di "Riavvia
    // pipeline" nel mock / "Restart" nella GUI Tkinter.
    state.hasPlayer = false;
    await applySettings();
  }

  async function onPlayPause() {
    if (state.playing) {
      state.playing = false;
      setPlayIcon(false);
      setStatusPill("Paused", "idle");
      try { await api().pause(); } catch (err) { /* window closing, etc. */ }
      return;
    }
    const ok = await ensurePlayerBuilt();
    if (!ok) return;
    state.playing = true;
    setPlayIcon(true);
    setStatusPill("Playing", "live");
    const speed = parseFloat($("speed-select").value) || 1;
    const sourceFps = parseFloat($("fps-input").value) || 15;
    try {
      await api().play(sourceFps * speed);
    } catch (err) {
      state.playing = false;
      setPlayIcon(false);
      setStatusPill(String(err.message || err), "idle");
    }
  }

  async function seekOrCatchUp(targetIndex) {
    // Salta istantaneamente se il frame e' gia' nella cache elaborata,
    // altrimenti elabora in sequenza fino a raggiungerlo (nessun salto
    // impossibile in avanti) -- stessa logica riusata da timeline/skip.
    targetIndex = Math.max(0, targetIndex);
    state.playing = false;
    setPlayIcon(false);
    try {
      const payload = await api().seek(targetIndex);
      if (payload.ok === false) {
        setStatusPill("Catching up…", "idle");
        while (state.cachedFrameCount <= targetIndex) {
          const step = await api().step_forward();
          window.onPipelineFrame(step);
          if (step.status && step.status.is_finished) break;
        }
      } else {
        window.onPipelineFrame(payload);
      }
    } catch (err) {
      setStatusPill(String(err.message || err), "idle");
    }
  }

  async function onSkipBack() {
    if (!state.hasPlayer) return;
    const fps = parseFloat($("fps-input").value) || 15;
    await seekOrCatchUp(state.lastKnownCursor - Math.round(10 * fps));
  }

  async function onSkipForward() {
    const ok = await ensurePlayerBuilt();
    if (!ok) return;
    const fps = parseFloat($("fps-input").value) || 15;
    await seekOrCatchUp(state.lastKnownCursor + Math.round(10 * fps));
  }

  async function onTimelineClick(evt) {
    if (!state.hasPlayer) return;
    const track = $("timeline-track");
    const rect = track.getBoundingClientRect();
    const ratio = Math.min(1, Math.max(0, (evt.clientX - rect.left) / rect.width));
    // Il click e' proporzionale alla durata TOTALE nota (se disponibile),
    // non solo al prefisso gia' elaborato -- cliccare oltre il prefisso
    // avvia il recupero sequenziale (vedi seekOrCatchUp), non un salto.
    const referenceFrames = state.totalFrameCount || state.cachedFrameCount || 1;
    const targetIndex = Math.round(ratio * (referenceFrames - 1));
    await seekOrCatchUp(targetIndex);
  }

  async function onExportCsv() {
    try {
      const path = await api().pick_save_csv_path();
      if (!path) return;
      const result = await api().export_csv(path);
      setStatusPill(result.ok ? `Saved ${result.rows} rows` : (result.error || "Save failed"), "idle");
    } catch (err) {
      setStatusPill(String(err.message || err), "idle");
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
    $("btn-play-analysis").addEventListener("click", onPlayPause);
    $("btn-restart").addEventListener("click", onRestart);
    $("btn-play").addEventListener("click", onPlayPause);
    $("btn-forward").addEventListener("click", onSkipForward);
    $("btn-back").addEventListener("click", onSkipBack);
    $("btn-export-csv").addEventListener("click", onExportCsv);
    $("btn-fullscreen").addEventListener("click", onFullscreen);
    $("timeline-track").addEventListener("click", onTimelineClick);
    $("mode-select").addEventListener("change", applyModeGating);
    $("arch-select").addEventListener("change", applyArchGating);
    // qualunque cambio di parametro invalida il player corrente: bisogna
    // premere "Play analysis"/"Restart" per ricostruirlo (un tracker gia'
    // avviato non si puo' riconfigurare a meta' strada, vedi
    // video_player.py) -- qui ci limitiamo a segnalarlo, senza ricostruire
    // da soli ad ogni click.
    const invalidatingIds = [
      "arch-select", "mode-select", "scale-select", "max-people-input",
      "reid-toggle", "hands-toggle", "eyes-toggle", "mouth-toggle",
      "eyebrows-toggle", "head-movement-toggle", "mediapipe-pose-toggle",
      "sam-chunk-size-input", "sam-overlap-input",
    ];
    invalidatingIds.forEach((id) => {
      $(id).addEventListener("change", () => {
        state.hasPlayer = false;
        updatePipelineFlow();
      });
    });
  }

  async function init() {
    wireEvents();
    applyModeGating();  // chiama gia' applyArchGating(), vedi sopra
    updateTimeline();
    try {
      const result = await api().detect_device();
      state.detectedDevice = result && result.device;
    } catch (err) {
      // aperto in un browser normale senza pywebview (vedi api()), o
      // detect_device() ha fallito per qualche motivo: SAM 3.1/SAM2
      // restano disabilitati per sicurezza (cudaAvailable resta false).
      state.detectedDevice = null;
    }
    applyArchGating();
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
