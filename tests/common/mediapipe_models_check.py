"""
mediapipe_models_check.py
============================
Verifies `common/mediapipe_models.py::resolve_model_path()` -- the shared
resolution/auto-download logic for MediaPipe Tasks models used by
`pose/mediapipe_pose.py`, `pose/hands.py` and `pose/gaze_head.py`
(extracted from there to avoid tripling it) -- WITHOUT importing
`mediapipe` or making real network downloads (`_download` replaced by a
fake that only creates an empty file).

Born from a real bug observed by Michele on two different models
(pose_landmarker_lite.task, then hand_landmarker.task/face_landmarker.task
with the exact same issue): launching the pipeline from a cwd different
from the one where the file had been downloaded by hand, MediaPipe would
fail with "unable to find <model name>" -- cause: the default was the
bare file name, resolved by MediaPipe as a path RELATIVE TO THE CWD.

Usage:
    python tests/mediapipe_models_check.py
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import common.mediapipe_models as mm  # noqa: E402

_FAKE_URL = "https://example.invalid/models/fake_landmarker/float16/1/fake_landmarker.task"
_FAKE_BASENAME = "fake_landmarker.task"


def part1_existing_explicit_path_used_as_is():
    # A path the user passed EXPLICITLY and that already exists -- no
    # resolution/download, used as-is even if the name has nothing to do
    # with the default one.
    tmp_dir = tempfile.mkdtemp()
    try:
        custom_path = os.path.join(tmp_dir, "some_arbitrary_name.task")
        Path(custom_path).write_bytes(b"fake model bytes")
        result = mm.resolve_model_path(custom_path, download_url=_FAKE_URL)
        assert result == custom_path, f"an existing path must be returned unchanged, got {result}"
        print("PASS part1_existing_explicit_path_used_as_is")
    finally:
        shutil.rmtree(tmp_dir)


def part2_custom_missing_path_not_touched():
    # Custom path (name different from the bare default) that does NOT
    # exist: no attempt to guess/download -- returned unchanged, so the
    # caller (MediaPipe) fails with the original error, clearer than a
    # silent download in the wrong place.
    missing = "/nonexistent/path/some_arbitrary_name.task"
    result = mm.resolve_model_path(missing, download_url=_FAKE_URL)
    assert result == missing
    print("PASS part2_custom_missing_path_not_touched")


def part3_bare_default_name_reuses_existing_cache_file():
    tmp_dir = tempfile.mkdtemp()
    original_cache_dir = mm.MODELS_CACHE_DIR
    original_download = mm._download
    original_cwd = os.getcwd()
    # resolve_model_path() FIRST checks whether the bare name already
    # exists in the current cwd -- we move to a clean folder so as not to
    # depend on any files left over from previous runs.
    os.chdir(tmp_dir)
    download_calls = []
    mm._download = lambda url, dest: download_calls.append((url, dest))
    try:
        cache_dir = Path(tmp_dir) / "cache"
        cache_path = cache_dir / _FAKE_BASENAME
        cache_dir.mkdir()
        cache_path.write_bytes(b"already downloaded")
        mm.MODELS_CACHE_DIR = cache_dir

        result = mm.resolve_model_path(_FAKE_BASENAME, download_url=_FAKE_URL)
        assert result == str(cache_path)
        assert download_calls == [], "the file is already in the cache -- no download needed again"
        print("PASS part3_bare_default_name_reuses_existing_cache_file")
    finally:
        mm.MODELS_CACHE_DIR = original_cache_dir
        mm._download = original_download
        os.chdir(original_cwd)
        shutil.rmtree(tmp_dir)


def part4_bare_default_name_triggers_download_when_missing():
    tmp_dir = tempfile.mkdtemp()
    original_cache_dir = mm.MODELS_CACHE_DIR
    original_download = mm._download
    original_cwd = os.getcwd()
    os.chdir(tmp_dir)

    def fake_download(url: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"downloaded now")

    mm._download = fake_download
    try:
        cache_dir = Path(tmp_dir) / "cache"
        cache_path = cache_dir / _FAKE_BASENAME
        mm.MODELS_CACHE_DIR = cache_dir
        assert not cache_path.exists(), "precondition: the file must not exist yet"

        result = mm.resolve_model_path(_FAKE_BASENAME, download_url=_FAKE_URL)
        assert result == str(cache_path)
        assert cache_path.exists(), "the fake download should have created the file"
        print("PASS part4_bare_default_name_triggers_download_when_missing")
    finally:
        mm.MODELS_CACHE_DIR = original_cache_dir
        mm._download = original_download
        os.chdir(original_cwd)
        shutil.rmtree(tmp_dir)


def part5_default_basename_derived_from_url_not_hardcoded():
    # The "default" name isn't a fixed constant: it's derived from the
    # last piece of download_url -- so the same helper serves
    # pose_landmarker_lite.task, hand_landmarker.task AND
    # face_landmarker.task without needing to parametrize the name separately.
    other_url = "https://example.invalid/models/other/float16/1/other_landmarker.task"
    missing_other = "other_landmarker.task"  # bare, but for a URL different from _FAKE_URL
    result = mm.resolve_model_path(missing_other, download_url=_FAKE_URL)
    # missing_other's basename ("other_landmarker.task") does NOT match
    # _FAKE_URL's basename ("fake_landmarker.task") -- treated as a
    # missing custom path, not resolved to the cache.
    assert result == missing_other, (
        "a bare name that doesn't match download_url's basename must be treated as a "
        "custom path (not resolved/downloaded), not just because it 'looks like' a MediaPipe model"
    )
    del other_url  # only for clarity of the comment above, not used otherwise
    print("PASS part5_default_basename_derived_from_url_not_hardcoded")


def main():
    part1_existing_explicit_path_used_as_is()
    part2_custom_missing_path_not_touched()
    part3_bare_default_name_reuses_existing_cache_file()
    part4_bare_default_name_triggers_download_when_missing()
    part5_default_basename_derived_from_url_not_hardcoded()
    print("\nAll common/mediapipe_models.py tests (resolve_model_path, without mediapipe/network) "
          "passed.")


if __name__ == "__main__":
    main()
