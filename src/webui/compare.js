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
 * loaded via a file:// URL are enough -- no inference, no per-frame IPC,
 * no server-side cache to manage. Much simpler, and the browser's own
 * decoder handles seeking/playback natively.
 *
 * Sync model (confirmed with Michele: synced, not just "started
 * together"): the FIRST loaded slot (lowest index with a video) is the
 * "leader" -- its `timeupdate` event drives the shared scrub bar/time
 * display, and a periodic timer nudges every OTHER loaded video back to
 * the leader's `currentTime` if it has drifted more than
 * DRIFT_THRESHOLD_S (independent <video> elements decode/advance
 * slightly differently, so left unchecked they'd slowly desync during
 * playback). Play/Pause/Restart/seek always act on every loaded slot at
 * once, using the SAME absolute time in seconds for all of them (not a
 * proportional position) -- videos being compared are expected to be
 * different inference runs over the SAME source footage, so the same
 * timestamp is the meaningful point of comparison.
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
    loaded: [false, false, false, false],
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

  function leaderSlot() {
    for (let i = 0; i < SLOT_COUNT; i++) {
      if (state.loaded[i]) return i;
    }
    return -1;
  }

  function anyLoaded() { return state.loaded.some(Boolean); }

  function setStatus(text) {
    document.getElementById("compare-status").textContent = text;
  }

  function updateStatus() {
    const n = state.loaded.filter(Boolean).length;
    setStatus(n > 0
      ? `${n} video${n === 1 ? "" : "s"} loaded — press Play to compare in sync.`
      : "Load up to 4 videos, then press Play — they stay in sync.");
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
    if (!anyLoaded()) return;
    for (let i = 0; i < SLOT_COUNT; i++) {
      if (state.loaded[i]) videoEl(i).play().catch(() => {});
    }
    state.playing = true;
    setPlayIcon(true);
  }

  function pauseAll() {
    for (let i = 0; i < SLOT_COUNT; i++) {
      if (state.loaded[i]) videoEl(i).pause();
    }
    state.playing = false;
    setPlayIcon(false);
  }

  function togglePlayPause() {
    if (state.playing) pauseAll(); else playAll();
  }

  function restartAll() {
    for (let i = 0; i < SLOT_COUNT; i++) {
      if (state.loaded[i]) videoEl(i).currentTime = 0;
    }
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
    for (let i = 0; i < SLOT_COUNT; i++) {
      if (state.loaded[i]) videoEl(i).currentTime = target;
    }
    updateTimeline();
  }

  function driftCorrect() {
    const lead = leaderSlot();
    if (lead < 0 || !state.playing) return;
    const leadTime = videoEl(lead).currentTime;
    for (let i = 0; i < SLOT_COUNT; i++) {
      if (i === lead || !state.loaded[i]) continue;
      const el = videoEl(i);
      if (Math.abs(el.currentTime - leadTime) > DRIFT_THRESHOLD_S) {
        el.currentTime = leadTime;
      }
    }
  }

  // Turns a filesystem path (as returned by the native file dialog) into
  // a file:// URL a <video> element can load. Assumes forward-slash
  // paths (macOS/Linux, consistent with the rest of this project's
  // target platforms, see README) -- would need adjusting for Windows
  // backslash paths.
  function pathToFileUrl(path) {
    return "file://" + path.split("/").map(encodeURIComponent).join("/");
  }

  async function loadIntoSlot(slot) {
    try {
      const path = await api().pick_video_path();
      if (!path) return;
      const el = videoEl(slot);
      el.src = pathToFileUrl(path);
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
