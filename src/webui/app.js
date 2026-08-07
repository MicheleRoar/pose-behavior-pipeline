/*
 * app.js
 * =======
 * Logica del frontend "Behaviour Vision Lab": legge lo stato dei controlli
 * della sidebar (Input / Task / Segmentation / Pose / Identity &
 * Re-identification / Outputs / Advanced settings), chiama il bridge Python
 * (webui/api.py, esposto come `window.pywebview.api.<metodo>(...)`, ognuno
 * una promise), e aggiorna il DOM (frame video, timeline, status bar) in
 * risposta ai payload ricevuti.
 *
 * Segmentazione e pose sono scelte INDIPENDENTI (vedi #segmentation-card e
 * #pose-card): non piu' un'unica selezione "Architecture" che decideva
 * entrambe. Il backend (gui/pipeline_runner.py) sceglie il wiring esatto in
 * base alla combinazione Task/Segmentation-model/Pose-model -- questo file
 * si limita a mostrare/nascondere i controlli giusti e a spiegare in una
 * riga (vedi applySegGuidance/applyPoseGuidance) da dove viene l'input per
 * ciascuna combinazione, cosi' l'utente non deve indovinarlo.
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
    totalFrameCount: null,   // da probe_video_metadata via pick_video_file() -- solo informativo
    totalDurationS: null,
    lastTimecodeS: 0,
    maxPeople: null,
    detectedDevice: null,  // da Api.detect_device() (vedi init()) -- usato SOLO per
    // abilitare/disabilitare SAM 3.1/SAM2 nel selettore Segmentation model, il
    // controllo definitivo resta lato server in Api.build_player().
    torchreidAvailable: false,  // idem, ma per il toggle "Appearance embedding (OSNet)"
    task: "both",       // "segmentation" | "pose" | "both" -- vedi #task-segmented
    session: "multiple", // "single" | "multiple" -- vedi #session-segmented
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
      (playing ? glyphPause : glyphPlay) + (playing ? " Pause" : " Run analysis");
  }

  // ------------------------------------------------------- segmented control
  // Task (Segmentation/Pose/Both) e Session (Single/Multiple person) usano
  // lo stesso pattern: un gruppo di bottoni con data-value, una classe
  // "active" spostata al click, un callback per rifare il gating dipendente.
  function wireSegmented(containerId, onChange) {
    const container = $(containerId);
    container.querySelectorAll(".seg-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        container.querySelectorAll(".seg-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        onChange(btn.dataset.value);
      });
    });
  }

  function segmentedValue(containerId) {
    const active = $(containerId).querySelector(".seg-btn.active");
    return active ? active.dataset.value : null;
  }

  // ------------------------------------------------------------- gating
  // Task -> quali card (Segmentation/Pose) sono visibili, e ricalcola tutto
  // il resto a cascata (guidance, identity, outputs, summary) -- stesso
  // ruolo di applyModeGating() nella versione precedente, ma su un albero di
  // controlli piu' ricco.
  function applyTaskGating() {
    const task = state.task;
    $("segmentation-card").classList.toggle("hidden", task === "pose");
    $("pose-card").classList.toggle("hidden", task === "segmentation");
    applySegGuidance();
    applyPoseGuidance();
    applyOutputsGating();
    applyIdentityGating();
    applyEmbeddingGating();
    updateSummary();
  }

  // Segmentation model -> riga di guida sotto il selettore ("Guidance: text
  // prompt 'person'" ecc., vedi il mock), campo prompt SAM 3.1, gating CUDA
  // per SAM 3.1/SAM2 (SAMURAI resta SEMPRE disabilitato, vedi l'attributo
  // "disabled" nell'HTML e il commento li' sul perche': il suo filtro di
  // Kalman non regge il multi-persona).
  function applySegGuidance() {
    const select = $("seg-model-select");
    const cudaAvailable = state.detectedDevice === "cuda";
    const segCapable = state.task !== "pose";

    ["sam31", "sam2"].forEach((value) => {
      const opt = select.querySelector(`option[value="${value}"]`);
      if (opt) opt.disabled = !cudaAvailable;
    });

    const isSam = select.value === "sam31" || select.value === "sam2";
    if (isSam && !cudaAvailable) {
      select.value = "yolo";
    }
    $("seg-cuda-hint").classList.toggle("hidden", cudaAvailable);

    const guidance = {
      yolo: "No extra guidance needed — YOLO26 detects people on its own.",
      sam31: "Guidance: text prompt (e.g. “person”), or auto boxes from YOLO if left empty.",
      sam2: "Guidance: auto boxes from YOLO (SAM2 has no text-prompt option).",
    }[select.value] || "";
    $("seg-guidance-hint").textContent = segCapable ? guidance : "";

    $("sam31-text-prompt-field").classList.toggle("hidden", select.value !== "sam31" || !cudaAvailable);

    const showChunkFields = select.value === "sam31" || select.value === "sam2";
    $("sam-chunk-fields").classList.toggle("hidden", !showChunkFields);
  }

  // Pose model -> riga di guida: dipende sia dal modello di pose sia da
  // COSA sta alimentando MediaPipe in questa combinazione (vedi la tabella
  // di auto-selezione dell'input nel docstring di
  // gui/pipeline_runner.iter_pipeline_frames -- questa funzione la
  // rispecchia in linguaggio naturale, non la reinventa).
  function applyPoseGuidance() {
    const poseModel = $("pose-model-select").value;
    const segActive = state.task !== "pose";
    let text;
    if (poseModel === "yolo") {
      text = "Input: full frame (independent YOLO26 Pose detector + tracker).";
    } else if (segActive) {
      text = "Input: tracked person crops (from the segmentation mask/box).";
    } else {
      text = "Input: auto-detected person boxes (internal YOLO proposer, no mask shown).";
    }
    $("pose-guidance-hint").textContent = text;
    $("scale-select").parentElement; // no-op, model size applies to any YOLO model in play
  }

  // Identity & Re-identification: mostra/nasconde il campo "Max people"
  // (irrilevante con Session=Single, forzato a 1 comunque), abilita/
  // disabilita "Flag uncertain matches"/"Lost identity memory" (hanno senso
  // solo con Tracking + Re-identification), aggiorna la pillola di stato.
  //
  // La pillola deve rispecchiare ESATTAMENTE la stessa condizione usata lato
  // server (vedi webui/api.py::build_player_kwargs, variabile
  // `seg_reid_ready`): il motore di re-id sulla SEGMENTAZIONE
  // (SegReIdentifier) richiede max_people per costruzione (solleva
  // ValueError altrimenti), quello sulla POSE (ReIdentifier) invece no. Se
  // qui la mostrassimo sempre come "Re-ID active" appena e' selezionato il
  // menu, un utente con Task=Segmentation/Both, Session=Multiple e "Max
  // number of people" vuoto vedrebbe una pillola verde MENTRE la
  // segmentazione gira comunque senza nessuna riassociazione -- esattamente
  // il bug diagnosticato dagli screenshot di ID instabili (296/169/7 sulla
  // stessa persona): il campo sembrava attivo ma non lo era.
  function applyIdentityGating() {
    const mode = $("identity-mode-select").value;
    const isReid = mode === "tracking_reid";
    const isSingle = state.session === "single";
    const segInvolved = state.task !== "pose";

    $("max-people-row").classList.toggle("hidden", isSingle);
    $("flag-uncertain-toggle").disabled = !isReid;
    $("lost-memory-input").disabled = !isReid;

    const maxPeopleSet = $("max-people-input").value !== "";
    const segReidBlocked = isReid && segInvolved && !isSingle && !maxPeopleSet;
    $("max-people-input").classList.toggle("field-input-required", isReid && segInvolved && !isSingle && !maxPeopleSet);

    const pill = $("reid-status-pill");
    const label = $("reid-status-label");
    if (mode === "frame_by_frame") {
      label.textContent = "Frame-by-frame";
      pill.className = "pill";
    } else if (mode === "tracking_only") {
      label.textContent = "Tracking only";
      pill.className = "pill";
    } else if (segReidBlocked) {
      label.textContent = "Re-ID needs Max people (segmentation)";
      pill.className = "pill pill-warn";
    } else {
      label.textContent = "Re-ID active";
      pill.className = "pill pill-accent";
    }
  }

  // Outputs: mani/viso restano cablati SOLO sul percorso YOLO26 Pose (vedi
  // webui/api.py::build_player_kwargs) -- con MediaPipe selezionato in Pose
  // lo scheletro c'e' comunque, ma dita/blink/bocca/sopracciglia non sono
  // ancora collegati su quel percorso (limite onesto, vedi
  // pipeline_runner._iter_pose_mediapipe).
  function applyOutputsGating() {
    const poseCapable = state.task !== "segmentation";
    const yoloPose = $("pose-model-select").value === "yolo";
    const enabled = poseCapable && yoloPose;

    ["hands-toggle", "eyes-toggle", "mouth-toggle", "eyebrows-toggle", "head-movement-toggle"].forEach((id) => {
      $(id).disabled = !enabled;
      if (!enabled) $(id).checked = false;
    });
    $("outputs-card").classList.toggle("disabled", !poseCapable);

    if (!poseCapable) {
      $("outputs-hint").textContent = "Segmentation-only: no pose output selected.";
    } else if (!yoloPose) {
      $("outputs-hint").textContent = "Hands/face signals need Pose model = YOLO26 Pose.";
    } else {
      $("outputs-hint").textContent = "";
    }
  }

  // ---------------------------------------------------------------- summary
  // Riepilogo in linguaggio naturale della configurazione corrente, ricavato
  // dai controlli reali -- mai un testo fisso, cosi' non puo' disallinearsi
  // da cosa gira davvero (stesso principio di updatePipelineFlow(), che
  // resta invariata piu' sotto per lo status bar).
  function updateSummary() {
    const scale = $("scale-select").value;
    const lines = [];

    const segLabel = { yolo: `YOLO26${scale} Segment`, sam31: "SAM 3.1", sam2: "SAM2" }[$("seg-model-select").value];
    const poseLabel = $("pose-model-select").value === "mediapipe" ? "MediaPipe" : `YOLO26${scale} Pose`;

    if (state.task === "segmentation") {
      lines.push(`<strong>${segLabel} masks</strong>`);
    } else if (state.task === "pose") {
      lines.push(`<strong>${poseLabel} skeletons</strong>`);
    } else {
      lines.push(`<strong>${segLabel} masks + ${poseLabel} skeletons</strong>`);
    }

    const identityMode = $("identity-mode-select").value;
    const identityText = {
      frame_by_frame: "no persistent identity (frame-by-frame)",
      tracking_only: "tracking only, no re-identification",
      tracking_reid: "tracking with heuristic Re-ID",
    }[identityMode];
    lines.push(`Identity: ${identityText}`);

    if (identityMode === "tracking_reid" && $("flag-uncertain-toggle").checked) {
      lines.push("Uncertain matches will be flagged, not auto-merged.");
    }
    if (state.session === "single") {
      lines.push("Session: single person expected.");
    }

    const warnings = [];
    if (($("seg-model-select").value === "sam31" || $("seg-model-select").value === "sam2")
        && state.detectedDevice !== "cuda") {
      warnings.push("SAM 3.1/SAM2 need a CUDA GPU — will fall back to YOLO26 Segment.");
    }
    const segInvolved = state.task !== "pose";
    const maxPeopleSet = $("max-people-input").value !== "";
    if (identityMode === "tracking_reid" && segInvolved && state.session !== "single" && !maxPeopleSet) {
      warnings.push("Segmentation IDs will NOT be re-identified: set “Max number of people” "
        + "above, or switch Session to Single. Without it, every exit/re-entry gets a brand-new ID.");
    }
    warnings.forEach((w) => lines.push(`<span class="summary-warn">${w}</span>`));

    $("summary-panel").innerHTML = lines.map((l) => `<span class="summary-line">${l}</span>`).join("");
  }

  const FLOW_ICONS = {
    box: '<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 2 3 7v10l9 5 9-5V7l-9-5Z" stroke="white" stroke-width="1.8" stroke-linejoin="round"/><path d="M3 7l9 5 9-5" stroke="white" stroke-width="1.8" stroke-linejoin="round"/></svg>',
    target: '<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="8" stroke="white" stroke-width="1.8"/><circle cx="12" cy="12" r="2.4" fill="white"/></svg>',
    link: '<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M9 15l6-6M8 16l-2 2a3.5 3.5 0 0 1-5-5l2-2M16 8l2-2a3.5 3.5 0 0 1 5 5l-2 2" stroke="white" stroke-width="1.8" stroke-linecap="round"/></svg>',
    nodes: '<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="5" cy="7" r="2" fill="white"/><circle cx="19" cy="7" r="2" fill="white"/><circle cx="12" cy="18" r="2" fill="white"/><path d="M6.6 8.2 10.5 16M17.4 8.2 13.5 16M7 7h10" stroke="white" stroke-width="1.4"/></svg>',
  };

  function updatePipelineFlow() {
    // Diagramma di flusso REALE, costruito dalla configurazione corrente
    // della sidebar -- non un testo fisso, cosi' riflette sempre la
    // combinazione Task/Segmentation-model/Pose-model davvero selezionata.
    const scale = $("scale-select").value;
    const steps = [];

    if (state.task !== "pose") {
      const seg = $("seg-model-select").value;
      const segLabel = seg === "sam31" ? "SAM 3.1 Segment" : seg === "sam2" ? "SAM2 Segment" : `YOLO26${scale} Segment`;
      steps.push({ label: segLabel, icon: "box", cls: "seg" });
    }
    if (state.task !== "segmentation") {
      const poseLabel = $("pose-model-select").value === "mediapipe" ? "MediaPipe Pose" : `YOLO26${scale} Pose`;
      steps.push({ label: poseLabel, icon: "box", cls: "pose" });
    }
    steps.push({ label: "ByteTrack", icon: "target", cls: "track" });
    const identityMode = $("identity-mode-select").value;
    if (identityMode === "tracking_reid") {
      steps.push({ label: "Re-ID", icon: "link", cls: "reid" });
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

  // Appearance embedding (OSNet, Advanced settings): stesso schema di
  // applySegGuidance() per SAM 3.1/SAM2 -- disabilita la checkbox e mostra
  // il motivo se 'torch'/'torchreid' non sono installati (vedi
  // Api.detect_device()), invece di lasciar scattare un errore solo dopo
  // "Run analysis". Il controllo definitivo resta comunque lato server
  // (pose/appearance_embedding.OSNetEmbedder solleva ImportError se forzato
  // comunque, vedi pipeline_runner._build_embedder).
  function applyEmbeddingGating() {
    const toggle = $("appearance-embedding-toggle");
    if (!state.torchreidAvailable) {
      toggle.checked = false;
      toggle.disabled = true;
    } else {
      toggle.disabled = false;
    }
    $("embedding-unavailable-hint").classList.toggle("hidden", state.torchreidAvailable);
  }

  function refreshAllGating() {
    applySegGuidance();
    applyPoseGuidance();
    applyOutputsGating();
    applyIdentityGating();
    applyEmbeddingGating();
    updatePipelineFlow();
    updateSummary();
  }

  // ------------------------------------------------------------- collect
  function collectParams() {
    // Nessun campo "device": lo lasciamo assente cosi' Api.build_player()
    // lo auto-rileva lato Python (cuda/mps/cpu, vedi common/device.py).
    return {
      mode: state.task,
      fps: $("fps-input").value,
      scale: $("scale-select").value,
      seg_backend: $("seg-model-select").value,
      pose_backend: $("pose-model-select").value,
      identity_mode: $("identity-mode-select").value,
      session_mode: state.session,
      flag_uncertain: $("flag-uncertain-toggle").checked,
      lost_identity_memory_s: $("lost-memory-input").value,
      max_people: state.session === "single" ? "" : $("max-people-input").value,
      with_hands: $("hands-toggle").checked,
      with_eyes: $("eyes-toggle").checked,
      with_mouth: $("mouth-toggle").checked,
      with_eyebrows: $("eyebrows-toggle").checked,
      with_head_movement: $("head-movement-toggle").checked,
      sam_chunk_size: $("sam-chunk-size-input").value,
      sam_overlap: $("sam-overlap-input").value,
      sam_redetect_every: $("sam-redetect-every-input").value,
      sam_text_prompt: $("sam-text-prompt-input").value,
      conf_threshold: $("conf-threshold-input").value,
      tracker_config: $("tracker-select").value,
      use_appearance_embedding: $("appearance-embedding-toggle").checked,
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

    wireSegmented("task-segmented", (value) => { state.task = value; applyTaskGating(); });
    wireSegmented("session-segmented", (value) => { state.session = value; applyIdentityGating(); updateSummary(); });

    $("seg-model-select").addEventListener("change", () => { applySegGuidance(); refreshAllGating(); });
    $("pose-model-select").addEventListener("change", () => { applyPoseGuidance(); applyOutputsGating(); refreshAllGating(); });
    $("identity-mode-select").addEventListener("change", () => { applyIdentityGating(); refreshAllGating(); });
    $("scale-select").addEventListener("change", refreshAllGating);
    // "input" (non "change"): la pillola/il bordo rosso devono aggiornarsi
    // mentre l'utente digita il numero, non solo quando il campo perde il
    // focus -- altrimenti resterebbe "Re-ID needs Max people" per un
    // istante dopo che il valore e' gia' stato inserito.
    $("max-people-input").addEventListener("input", () => { applyIdentityGating(); updateSummary(); });

    // qualunque cambio di parametro invalida il player corrente: bisogna
    // premere "Run analysis"/"Restart" per ricostruirlo (un tracker gia'
    // avviato non si puo' riconfigurare a meta' strada, vedi
    // video_player.py) -- qui ci limitiamo a segnalarlo, senza ricostruire
    // da soli ad ogni click.
    const invalidatingIds = [
      "seg-model-select", "pose-model-select", "identity-mode-select", "scale-select",
      "max-people-input", "flag-uncertain-toggle", "lost-memory-input",
      "hands-toggle", "eyes-toggle", "mouth-toggle", "eyebrows-toggle", "head-movement-toggle",
      "sam-chunk-size-input", "sam-overlap-input", "sam-redetect-every-input", "sam-text-prompt-input",
      "conf-threshold-input", "tracker-select",
    ];
    invalidatingIds.forEach((id) => {
      $(id).addEventListener("change", () => {
        state.hasPlayer = false;
        updatePipelineFlow();
        updateSummary();
      });
    });
    $("task-segmented").addEventListener("click", () => { state.hasPlayer = false; });
    $("session-segmented").addEventListener("click", () => { state.hasPlayer = false; });
  }

  async function init() {
    wireEvents();
    applyTaskGating();  // chiama gia' applySegGuidance/applyPoseGuidance/applyOutputsGating/applyIdentityGating
    updateTimeline();
    try {
      const result = await api().detect_device();
      state.detectedDevice = result && result.device;
      state.torchreidAvailable = !!(result && result.torchreid_available);
    } catch (err) {
      // aperto in un browser normale senza pywebview (vedi api()), o
      // detect_device() ha fallito per qualche motivo: SAM 3.1/SAM2
      // restano disabilitati per sicurezza (cudaAvailable resta false),
      // idem l'embedding OSNet (torchreidAvailable resta false).
      state.detectedDevice = null;
      state.torchreidAvailable = false;
    }
    refreshAllGating();
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
