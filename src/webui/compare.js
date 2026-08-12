/**
 * compare.js
 * ===========
 * Logic for compare.html, the "compare up to 4 runs" second window
 * (Michele, 2026-08: wanted to load several already-exported annotated
 * videos -- see webui/api.py::Api.export_video -- side by side, press
 * ONE play button, and see how different backend/parameter combinations
 * behave on the same footage).
 *
 * Deliberately NOT built on VideoPlayer/the JPEG-frame-over-IPC pipeline
 * used by the main window (app.js): these are already-finished, already-
 * annotated MP4 files sitting on disk, so plain HTML5 <video> elements
 * are enough -- no inference, no per-frame IPC, no server-side cache to
 * manage. The browser's own decoder handles seeking/playback natively.
 * Videos are served over a tiny local HTTP server (webui/local_media_server.py),
 * NOT a file:// src -- see loadIntoSlot()'s comment for why.
 *
 * Sync model (confirmed with Michele: synced, not just "started
 * together"): the FIRST loaded-AND-PLAYABLE slot is the "leader" -- its
 * `timeupdate` event drives the shared scrub bar/time display, and a
 * periodic timer nudges every OTHER playable video back to the leader's
 * `currentTime` if it has drifted more than DRIFT_THRESHOLD_S
 * (independent <video> elements decode/advance slightly differently, so
 * left unchecked they'd slowly desync during playback). Play/Pause/
 * Restart/seek always act on every playable slot at once, using the SAME
 * absolute time in seconds for all of them (not a proportional position)
 * -- videos being compared are expected to be different inference runs
 * over the SAME source footage, so the same timestamp is the meaningful
 * point of comparison.
 *
 * "loaded" vs "playable" (Michele, 2026-08, real Linux/CUDA machine: all
 * 4 slots showed "loaded", but Play did nothing and no thumbnail ever
 * appeared -- root cause was the exported MP4s using a codec ('mp4v'
 * fourcc = MPEG-4 Part 2, see common/video_writer.py) that Linux's
 * GStreamer-based decoders don't support, so the file loaded fine but
 * could never actually be decoded). These are now tracked separately:
 * `state.loaded[slot]` means "a file was picked and a src was assigned";
 * `state.errored[slot]` means "the browser tried and failed to decode
 * it" (surfaced via the <video>'s `error` event, see wireVideoEvents).
 * Only loaded-and-not-errored slots are "playable" -- see playableSlots()
 * -- and only those participate in play/pause/seek/sync. An errored slot
 * shows the real MediaError reason in its cell instead of silently doing
 * nothing, so this is diagnosable next time instead of just "nothing
 * happens".
 *
 * Known limitation (acceptable for v1, not requested): if a video is
 * loaded into an EARLIER slot after playback already started in a later
 * one, the leader silently shifts to the earlier slot and the timeline
 * can jump -- there's no "swap leader" UI. Loading all 4 before pressing
 * Play avoids this entirely.
 */

(() => {
  "use strict";

  const SLOT_COUNT = 4;
  const DRIFT_THRESHOLD_S = 0.15; // below the ~0.1s granularity most codecs seek to anyway
  const RESYNC_INTERVAL_MS = 300;

  const state = {
    loaded: [false, false, false, false],  // a file was picked, src assigned
    errored: [false, false, false, false], // the browser failed to decode it (see wireVideoEvents)
    playing: false,
  };

  function api() {
    if (!window.pywebview || !window.pywebview.api) {
      throw new Error("pywebview bridge not available (opened outside the app window?)");
    }
    return window.pywebview.api;
  }

  function videoEl(slot) { return document.getElementById(`video-${slot}`); }
  function wrapEl(slot) { return videoEl(slot).closest(".compare-video-wrap"); }
  function placeholderEl(slot) { return document.getElementById(`placeholder-${slot}`); }

  function isPlayable(slot) { return state.loaded[slot] && !state.errored[slot]; }

  function playableSlots() {
    const out = [];
    for (let i = 0; i < SLOT_COUNT; i++) if (isPlayable(i)) out.push(i);
    return out;
  }

  function leaderSlot() {
    const slots = playableSlots();
    return slots.length ? slots[0] : -1;
  }

  function setStatus(text) {
    document.getElementById("compare-status").textContent = text;
  }

  function updateStatus() {
    const nLoaded = state.loaded.filter(Boolean).length;
    const nErrored = state.errored.filter(Boolean).length;
    const nPlayable = nLoaded - nErrored;
    if (nErrored > 0) {
      setStatus(`${nPlayable}/${nLoaded} video${nLoaded === 1 ? "" : "s"} playable — `
        + `${nErrored} failed to decode, see the red message in that cell.`);
    } else if (nLoaded > 0) {
      setStatus(`${nLoaded} video${nLoaded === 1 ? "" : "s"} loaded — press Play to compare in sync.`);
    } else {
      setStatus("Load up to 4 videos, then press Play — they stay in sync.");
    }
  }

  function formatTime(seconds) {
    const s = Math.max(0, Math.floor(seconds || 0));
    const m = Math.floor(s / 60);
    const rem = s % 60;
    return `${String(m).padStart(2, "0")}:${String(rem).padStart(2, "0")}`;
  }

  function leaderDuration() {
    const i = leaderSlot();
    if (i < 0) return 0;
    const d = videoEl(i).duration;
    return isFinite(d) && d > 0 ? d : 0;
  }

  function updateTimeline() {
    const i = leaderSlot();
    const duration = leaderDuration();
    const current = i >= 0 ? videoEl(i).currentTime : 0;
    const pct = duration > 0 ? Math.min(100, (current / duration) * 100) : 0;
    document.getElementById("compare-timeline-progress").style.width = pct + "%";
    document.getElementById("compare-timecode").textContent =
      `${formatTime(current)} / ${formatTime(duration)}`;
  }

  function setPlayIcon(isPlaying) {
    document.getElementById("icon-play").classList.toggle("hidden", isPlaying);
    document.getElementById("icon-pause").classList.toggle("hidden", !isPlaying);
  }

  function playAll() {
    const slots = playableSlots();
    if (!slots.length) return;
    for (const i of slots) videoEl(i).play().catch(() => {});
    state.playing = true;
    setPlayIcon(true);
  }

  function pauseAll() {
    for (const i of playableSlots()) videoEl(i).pause();
    state.playing = false;
    setPlayIcon(false);
  }

  function togglePlayPause() {
    if (state.playing) pauseAll(); else playAll();
  }

  function restartAll() {
    for (const i of playableSlots()) videoEl(i).currentTime = 0;
    pauseAll();
    updateTimeline();
  }

  function onTimelineClick(evt) {
    const duration = leaderDuration();
    if (duration <= 0) return;
    const track = document.getElementById("compare-timeline-track");
    const rect = track.getBoundingClientRect();
    const ratio = Math.min(1, Math.max(0, (evt.clientX - rect.left) / rect.width));
    const target = ratio * duration;
    for (const i of playableSlots()) videoEl(i).currentTime = target;
    updateTimeline();
  }

  function driftCorrect() {
    const lead = leaderSlot();
    if (lead < 0 || !state.playing) return;
    const leadTime = videoEl(lead).currentTime;
    for (const i of playableSlots()) {
      if (i === lead) continue;
      const el = videoEl(i);
      if (Math.abs(el.currentTime - leadTime) > DRIFT_THRESHOLD_S) {
        el.currentTime = leadTime;
      }
    }
  }

  // Human-readable text for the standard HTMLMediaElement error codes --
  // https://developer.mozilla.org/en-US/docs/Web/API/MediaError/code --
  // MEDIA_ERR_SRC_NOT_SUPPORTED (4) is by far the most likely one here
  // (wrong/unsupported codec, see the module docstring's "loaded vs
  // playable" note), MEDIA_ERR_NETWORK (2) would mean the local HTTP
  // server (local_media_server.py) wasn't reachable or the file moved.
  function mediaErrorText(mediaError) {
    if (!mediaError) return "Unknown playback error.";
    switch (mediaError.code) {
      case 1: return "Load aborted.";
      case 2: return "Network error while loading the video (local server unreachable?).";
      case 3: return "The video data is corrupted and could not be decoded.";
      case 4: return "This video's codec isn't supported by this system's video player "
        + "(likely an H.264/.mp4 or old mpeg4 file -- re-export it: this project now "
        + "prefers VP9/.webm, which plays here without extra codec installs, see "
        + "common/video_writer.py).";
      default: return "Unknown playback error.";
    }
  }

  function markSlotError(slot, text) {
    state.errored[slot] = true;
    const ph = placeholderEl(slot);
    ph.textContent = text;
    ph.classList.add("is-error");
    updateStatus();
  }

  function clearSlotError(slot) {
    if (!state.errored[slot]) return;
    state.errored[slot] = false;
    const ph = placeholderEl(slot);
    ph.classList.remove("is-error");
    ph.textContent = "No video loaded";
    updateStatus();
  }

  async function loadIntoSlot(slot) {
    try {
      // pick_video_path() returns {"path": ..., "url": "http://127.0.0.1:.../..."}
      // -- NOT a plain file:// URL (BUG, Michele 2026-08, real Linux
      // machine: "Not allowed to load local resource" -- WebKitGTK/Qt,
      // pywebview's Linux backends, refuse to load a file:// resource
      // from a page itself served over file://, unlike macOS's
      // WKWebView). Python serves the picked file over a tiny local
      // HTTP server instead (see webui/local_media_server.py), which
      // works identically on every platform/backend.
      const result = await api().pick_video_path();
      if (!result) return;
      clearSlotError(slot);
      const el = videoEl(slot);
      el.src = result.url;
      el.load();
      state.loaded[slot] = true;
      wrapEl(slot).classList.add("has-video");
      updateStatus();
    } catch (err) {
      setStatus(String(err.message || err));
    }
  }

  function wireVideoEvents(slot) {
    const el = videoEl(slot);
    el.addEventListener("loadedmetadata", () => {
      clearSlotError(slot);
      updateTimeline();
      // Joining a slot mid-playback (loaded while the others are
      // already running): jump it to the current leader time and start
      // it too, instead of leaving it sitting at 0:00 out of sync.
      if (state.playing) {
        const lead = leaderSlot();
        if (lead >= 0 && lead !== slot) el.currentTime = videoEl(lead).currentTime;
        el.play().catch(() => {});
      }
    });
    el.addEventListener("timeupdate", updateTimeline);
    el.addEventListener("ended", () => {
      // One video reaching its end (they can have slightly different
      // lengths) stops the whole comparison rather than letting the
      // others keep going solo -- restart/play again to re-sync.
      pauseAll();
    });
    // The failure mode that motivated all of this (see module docstring):
    // a video whose src is set but that the browser can't actually
    // decode fires `error`, not `loadedmetadata` -- previously nothing
    // listened for this, so a bad codec just looked like "loaded, but
    // Play silently does nothing". Now it's surfaced in that cell AND
    // excluded from play/pause/seek/sync so it can't silently stall the
    // whole comparison.
    el.addEventListener("error", () => {
      markSlotError(slot, mediaErrorText(el.error));
    });
  }

  function wireEvents() {
    document.querySelectorAll("[data-slot]").forEach((btn) => {
      btn.addEventListener("click", () => loadIntoSlot(parseInt(btn.dataset.slot, 10)));
    });
    for (let i = 0; i < SLOT_COUNT; i++) wireVideoEvents(i);
    document.getElementById("btn-play-pause").addEventListener("click", togglePlayPause);
    document.getElementById("btn-restart").addEventListener("click", restartAll);
    document.getElementById("compare-timeline-track").addEventListener("click", onTimelineClick);
    setInterval(driftCorrect, RESYNC_INTERVAL_MS);
  }

  document.addEventListener("DOMContentLoaded", wireEvents);
})();
