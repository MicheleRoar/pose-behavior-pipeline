/*
 * app.js
 * =======
 * "Behaviour Vision Lab" frontend logic: reads the state of the sidebar
 * controls (Input / Task / Segmentation / Pose / Identity &
 * Re-identification / Outputs / Advanced settings), calls the Python
 * bridge (webui/api.py, exposed as `window.pywebview.api.<method>(...)`,
 * each one a promise), and updates the DOM (video frame, timeline, status
 * bar) in response to the received payloads.
 *
 * Segmentation and pose are INDEPENDENT choices (see #segmentation-card
 * and #pose-card): no longer a single "Architecture" selection that
 * decided both. The backend (gui/pipeline_runner.py) picks the exact
 * wiring based on the Task/Segmentation-model/Pose-model combination --
 * this file just shows/hides the right controls and explains in one
 * line (see applySegGuidance/applyPoseGuidance) where the input for each
 * combination comes from, so the user doesn't have to guess.
 *
 * Two different paths for getting a frame onto the screen:
 *  1. Play: `api.play(fps)` starts a background Python thread that pushes
 *     every frame on its own by calling `window.onPipelineFrame(payload)`
 *     (see webui/api.py::Api._push_frame) -- there's no JS-side polling
 *     here.
 *  2. Back/Forward/Seek: the call itself (`api.step_forward()` etc.)
 *     RETURNS the frame's payload, so it's passed directly to
 *     `onPipelineFrame` instead of waiting for a separate push.
 * In both cases the payload has the same shape {ok, frame, status}, so
 * rendering has a single entry point.
 */

(() => {
  "use strict";

  const state = {
    playing: false,
    hasPlayer: false,
    lastKnownCursor: 0,
    cachedFrameCount: 0,
    totalFrameCount: null,   // from probe_video_metadata via pick_video_file() -- informational only
    totalDurationS: null,
    lastTimecodeS: 0,
    maxPeople: null,
    detectedDevice: null,  // from Api.detect_device() (see init()) -- used ONLY to
    // enable/disable SAM 3.1/SAM2 in the Segmentation model selector, the
    // definitive check stays server-side in Api.build_player().
    torchreidAvailable: false,  // same idea, but for the "Appearance embedding (OSNet)" toggle
    task: "both",       // "segmentation" | "pose" | "both" -- see #task-segmented
    session: "multiple", // "single" | "multiple" -- see #session-segmented
    seekToken: 0,  // bumped on every seekOrCatchUp() call -- lets an older, still
    // in-flight catch-up loop notice it's been superseded and bail out instead of
    // continuing to fire step_forward() and painting stale frames over a newer
    // seek (see seekOrCatchUp; the backend itself is now race-safe too, see
    // webui/api.py's _advance()/seek() docstrings -- this is a UX/waste fix on
    // top of that, not a substitute for it).
  };

  // ---------------------------------------------------------------- utils
  function $(id) { return document.getElementById(id); }

  function api() {
    // window.pywebview.api isn't guaranteed to be ready until the
    // 'pywebviewready' event has fired (see bindReady below) -- but even
    // after that, if the module is opened in a regular browser for
    // debugging (without pywebview) the object simply doesn't exist:
    // here we flag it clearly instead of silently failing every button.
    if (!window.pywebview || !window.pywebview.api) {
      throw new Error("pywebview bridge not available (opened outside the app window?)");
    }
    return window.pywebview.api;
  }

  function setStatusPill(text, kind) {
    // "kind" is "live" (playing, dot lit) or "idle" (dot off).
    $("status-label").textContent = text;
    $("dot-status").className = "status-dot " + (kind === "live" ? "status-dot-live" : "status-dot-idle");
  }

  function updateDeviceIndicator(device) {
    // BUG (Michele, 2026-08): the badge text was hardcoded "GPU" in
    // index.html and NEVER updated -- always showed "GPU" even on a
    // Mac with no CUDA (mps/cpu). Now driven by the real device string
    // from detect_device()/status.device ("cuda"|"mps"|"cpu", see
    // common/device.py), both at startup and once a run is actually
    // configured.
    if (!device) return;
    const label = device === "cuda" ? "GPU" : device === "mps" ? "MPS" : "CPU";
    $("gpu-indicator-label").textContent = label;
    $("gpu-indicator").title = `Configured device: ${device}`;
    $("dot-device").className = "status-dot " + (device === "cuda" ? "status-dot-live" : "status-dot-idle");
  }

  function formatTimecode(seconds) {
    // "HH:MM:SS.cc" (hundredths), same format as the mock ("00:00:06.30").
    const s = Math.max(0, seconds || 0);
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const rem = s % 60;
    const cc = Math.round((rem - Math.floor(rem)) * 100);
    return [h, m, Math.floor(rem)].map((v) => String(v).padStart(2, "0")).join(":")
      + "." + String(cc).padStart(2, "0");
  }

  function formatTimeShort(seconds) {
    // "MM:SS" for the timeline ticks (or "H:MM:SS" if needed).
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
  // Task (Segmentation/Pose/Both) and Session (Single/Multiple person) use
  // the same pattern: a group of buttons with data-value, an "active"
  // class moved on click, a callback to redo the dependent gating.
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
  // Task -> which cards (Segmentation/Pose) are visible, and recomputes
  // everything else in a cascade (guidance, identity, outputs, summary)
  // -- same role as applyModeGating() in the previous version, but on a
  // richer control tree.
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

  // Segmentation model -> guidance line under the selector ("Guidance:
  // text prompt 'person'" etc., see the mock), SAM 3.1 prompt field, CUDA
  // gating for SAM 3.1/SAM2 (SAMURAI stays ALWAYS disabled, see the
  // "disabled" attribute in the HTML and the comment there on why: its
  // Kalman filter can't handle multiple people).
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

    // "Tracking method" (ByteTrack config) only affects the YOLO26
    // Segment backend (fed into Ultralytics' model.track(tracker=...),
    // see segmentation_demo.py::build_tracker) -- SAM 3.1/SAM2 never
    // read this value (their own chunked video propagation + id
    // reconciliation handles temporal continuity instead), so showing
    // an apparently-live selector there was misleading (2026-08 bug,
    // found by Michele). Hidden for those two backends, matching how
    // sam-chunk-fields/sam31-text-prompt-field already toggle above.
    $("tracker-select-field").classList.toggle("hidden", isSam);
  }

  // Pose model -> guidance line: depends both on the pose model and on
  // WHAT is feeding MediaPipe in this combination (see the input
  // auto-selection table in the docstring of
  // gui/pipeline_runner.iter_pipeline_frames -- this function mirrors it
  // in natural language, it doesn't reinvent it).
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

  // Identity & Re-identification: shows/hides the "Max people" field
  // (irrelevant with Session=Single, forced to 1 regardless), enables/
  // disables "Flag uncertain matches"/"Lost identity memory" (only make
  // sense with Tracking + Re-identification), updates the status pill.
  //
  // The pill must EXACTLY mirror the same condition used server-side
  // (see webui/api.py::build_player_kwargs, `seg_reid_ready` variable):
  // the re-id engine for SEGMENTATION (SegReIdentifier) requires
  // max_people by construction (raises ValueError otherwise), the one
  // for POSE (ReIdentifier) doesn't. If we always showed it here as
  // "Re-ID active" as soon as the menu is selected, a user with
  // Task=Segmentation/Both, Session=Multiple and an empty "Max number of
  // people" would see a green pill WHILE segmentation runs anyway with
  // no re-association at all -- exactly the bug diagnosed from the
  // unstable-ID screenshots (296/169/7 on the same person): the field
  // looked active but wasn't.
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

  // Outputs: hands/face stay wired ONLY to the YOLO26 Pose path (see
  // webui/api.py::build_player_kwargs) -- with MediaPipe selected for
  // Pose the skeleton is still there, but fingers/blink/mouth/eyebrows
  // aren't wired up on that path yet (honest limitation, see
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
  // Plain-language recap of the current configuration, derived from the
  // real controls -- never a fixed string, so it can never drift out of
  // sync with what's actually running (same principle as
  // updatePipelineFlow(), which stays unchanged further below for the
  // status bar).
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
    // REAL flow diagram, built from the sidebar's current configuration
    // -- not a fixed string, so it always reflects the Task/Segmentation-
    // model/Pose-model combination actually selected.
    const scale = $("scale-select").value;
    const steps = [];

    const seg = $("seg-model-select").value;
    const segIsSam = state.task !== "pose" && (seg === "sam31" || seg === "sam2");
    if (state.task !== "pose") {
      const segLabel = seg === "sam31" ? "SAM 3.1 Segment" : seg === "sam2" ? "SAM2 Segment" : `YOLO26${scale} Segment`;
      steps.push({ label: segLabel, icon: "box", cls: "seg" });
    }
    if (state.task !== "segmentation") {
      const poseLabel = $("pose-model-select").value === "mediapipe" ? "MediaPipe Pose" : `YOLO26${scale} Pose`;
      steps.push({ label: poseLabel, icon: "box", cls: "pose" });
    }
    // ByteTrack only runs for the YOLO26 Segment backend (Ultralytics
    // model.track()) -- SAM 3.1/SAM2 handle temporal continuity
    // themselves via chunked video propagation + id reconciliation
    // (segmentation/chunking.py), no ByteTrack involved. Showing
    // "ByteTrack" unconditionally here was misleading (2026-08 bug,
    // found by Michele: the label appeared even with SAM selected).
    if (segIsSam) {
      steps.push({ label: "SAM tracking (chunked)", icon: "target", cls: "track" });
    } else {
      steps.push({ label: "ByteTrack", icon: "target", cls: "track" });
    }
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

  // Appearance embedding (OSNet, Advanced settings): same pattern as
  // applySegGuidance() for SAM 3.1/SAM2 -- disables the checkbox and
  // shows the reason if 'torch'/'torchreid' aren't installed (see
  // Api.detect_device()), instead of letting an error surface only after
  // "Run analysis". The definitive check still stays server-side
  // (pose/appearance_embedding.OSNetEmbedder raises ImportError if
  // forced anyway, see pipeline_runner._build_embedder).
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
    // No "device" field: we leave it absent so Api.build_player()
    // auto-detects it server-side (cuda/mps/cpu, see common/device.py).
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
      updateDeviceIndicator(status.device);
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
    // If the total duration is known (read from the file's metadata at
    // load time, see api.py::probe_video_metadata) the bar shows "how
    // much of the whole video has been processed so far"; otherwise
    // (rare containers without this metadata) it falls back to "how much
    // of the current cache has been seen", which is all that can be
    // known anyway. In BOTH cases the scrubber remains clickable ONLY on
    // the already-processed prefix -- see onTimelineClick.
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
    // "Restart pipeline": forced rebuild from scratch, even if a player
    // already existed -- a tracker already started can't be reconfigured
    // halfway through (see gui/video_player.py), same role as "Restart
    // pipeline" in the mock / "Restart" in the Tkinter GUI.
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
    // Jumps instantly if the frame is already in the processed cache,
    // otherwise processes sequentially until reaching it (no impossible
    // forward jump) -- same logic reused by timeline/skip.
    targetIndex = Math.max(0, targetIndex);
    const myToken = ++state.seekToken;
    state.playing = false;
    setPlayIcon(false);
    try {
      const payload = await api().seek(targetIndex);
      if (state.seekToken !== myToken) return;  // a newer seek/click already took over
      if (payload.ok === false) {
        setStatusPill("Catching up…", "idle");
        while (state.cachedFrameCount <= targetIndex) {
          if (state.seekToken !== myToken) return;
          const step = await api().step_forward();
          if (state.seekToken !== myToken) return;  // superseded while awaiting
          window.onPipelineFrame(step);
          if (step.status && step.status.is_finished) break;
        }
      } else {
        window.onPipelineFrame(payload);
      }
    } catch (err) {
      if (state.seekToken === myToken) setStatusPill(String(err.message || err), "idle");
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
    // The click is proportional to the known TOTAL duration (if
    // available), not just the already-processed prefix -- clicking
    // beyond the prefix starts sequential catch-up (see seekOrCatchUp),
    // not a jump.
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

  async function onCompareRuns() {
    // Opens the "compare up to 4 runs" second window (Michele, 2026-08)
    // -- see webui/api.py::Api.open_compare_window()/compare.html. Just
    // a fire-and-forget call: the new window is self-contained
    // (compare.js), nothing to wire back here.
    try {
      await api().open_compare_window();
    } catch (err) {
      setStatusPill(String(err.message || err), "idle");
    }
  }

  async function onExportVideo() {
    // Saves the annotated video (overlay already drawn on every
    // processed frame) so different runs/parameter choices can be
    // compared side by side later -- same pick-then-save pattern as
    // onExportCsv, see webui/api.py::Api.export_video.
    try {
      const path = await api().pick_save_video_path();
      if (!path) return;
      setStatusPill("Saving video…", "idle");
      const result = await api().export_video(path);
      setStatusPill(result.ok ? `Saved ${result.frames} frames` : (result.error || "Save failed"), "idle");
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
    $("btn-export-video").addEventListener("click", onExportVideo);
    $("btn-compare-runs").addEventListener("click", onCompareRuns);
    $("btn-fullscreen").addEventListener("click", onFullscreen);
    $("timeline-track").addEventListener("click", onTimelineClick);

    wireSegmented("task-segmented", (value) => { state.task = value; applyTaskGating(); });
    wireSegmented("session-segmented", (value) => { state.session = value; applyIdentityGating(); updateSummary(); });

    $("seg-model-select").addEventListener("change", () => { applySegGuidance(); refreshAllGating(); });
    $("pose-model-select").addEventListener("change", () => { applyPoseGuidance(); applyOutputsGating(); refreshAllGating(); });
    $("identity-mode-select").addEventListener("change", () => { applyIdentityGating(); refreshAllGating(); });
    $("scale-select").addEventListener("change", refreshAllGating);
    // "input" (not "change"): the pill/red border must update while the
    // user is typing the number, not only when the field loses focus --
    // otherwise it would stay "Re-ID needs Max people" for a moment
    // after the value has already been entered.
    $("max-people-input").addEventListener("input", () => { applyIdentityGating(); updateSummary(); });

    // any parameter change invalidates the current player: "Run
    // analysis"/"Restart" must be pressed to rebuild it (a tracker
    // already started can't be reconfigured halfway through, see
    // video_player.py) -- here we just flag it, without rebuilding on
    // our own on every click.
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
    applyTaskGating();  // already calls applySegGuidance/applyPoseGuidance/applyOutputsGating/applyIdentityGating
    updateTimeline();
    try {
      const result = await api().detect_device();
      state.detectedDevice = result && result.device;
      state.torchreidAvailable = !!(result && result.torchreid_available);
      updateDeviceIndicator(state.detectedDevice);
    } catch (err) {
      // opened in a regular browser without pywebview (see api()), or
      // detect_device() failed for some reason: SAM 3.1/SAM2 stay
      // disabled for safety (cudaAvailable stays false), same for the
      // OSNet embedding (torchreidAvailable stays false).
      state.detectedDevice = null;
      state.torchreidAvailable = false;
    }
    refreshAllGating();
  }

  if (window.pywebview) {
    init();
  } else {
    // during development the file can be opened in a regular browser to
    // quickly check the layout: the controls work, the API calls fail
    // with a clear message instead of a silent error (see api()).
    document.addEventListener("DOMContentLoaded", init);
  }
  window.addEventListener("pywebviewready", init);
})();
