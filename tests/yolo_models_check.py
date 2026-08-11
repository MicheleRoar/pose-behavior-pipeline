"""
yolo_models_check.py
======================
Verifies `common/yolo_models.py`'s path resolution helpers -- no
ultralytics/torch needed (these functions only touch the filesystem).

Run with: python tests/yolo_models_check.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from common.yolo_models import CONFIGS_DIR, resolve_tracker_config, resolve_yolo_weights


def part1_configs_dir_points_at_real_src_configs():
    # Sanity check for the parents[N] arithmetic itself: get this wrong
    # and every other part below would "pass" against the wrong
    # directory without ever exercising the real bug.
    assert CONFIGS_DIR.is_dir(), f"expected {CONFIGS_DIR} to exist"
    assert (CONFIGS_DIR / "bytetrack_permissive.yaml").is_file(), (
        f"expected src/configs/bytetrack_permissive.yaml under {CONFIGS_DIR}"
    )
    print("Part 1: CONFIGS_DIR resolves to the real src/configs/ folder — OK")


def part2_bare_tracker_name_is_left_untouched():
    # "bytetrack.yaml"/"botsort.yaml" with no directory component are
    # Ultralytics' OWN bundled configs -- must not be redirected.
    assert resolve_tracker_config("bytetrack.yaml") == "bytetrack.yaml"
    assert resolve_tracker_config("botsort.yaml") == "botsort.yaml"
    print("Part 2: a bare tracker name (Ultralytics' own bundled config) is left unchanged — OK")


def part3_configs_relative_path_resolves_regardless_of_cwd():
    # Reproduces the real bug (Michele, 2026-08, Linux/CUDA machine):
    # "configs/bytetrack_permissive.yaml" is what the GUI sends (see
    # webui/index.html); Ultralytics would resolve that against the
    # CURRENT WORKING DIRECTORY, which fails unless the app happens to
    # be launched from src/. Run from a cwd that does NOT contain a
    # "configs/" folder at all, to prove this doesn't depend on cwd.
    original_cwd = os.getcwd()
    os.chdir(str(Path(__file__).resolve().parent))  # tests/ has no configs/ subfolder
    try:
        assert not os.path.isfile("configs/bytetrack_permissive.yaml"), (
            "test setup assumption broken: configs/bytetrack_permissive.yaml "
            "unexpectedly resolves from tests/ -- pick a different cwd for this check"
        )
        resolved = resolve_tracker_config("configs/bytetrack_permissive.yaml")
    finally:
        os.chdir(original_cwd)
    assert resolved == str(CONFIGS_DIR / "bytetrack_permissive.yaml"), resolved
    assert os.path.isfile(resolved), f"resolved path must actually exist: {resolved}"
    print("Part 3: 'configs/bytetrack_permissive.yaml' resolves to the real file "
          "regardless of the current working directory — OK")


def part4_already_existing_explicit_path_is_left_untouched():
    real_path = str(CONFIGS_DIR / "bytetrack.yaml")
    assert resolve_tracker_config(real_path) == real_path
    print("Part 4: an explicit path that already exists is returned unchanged — OK")


def part5_unresolvable_path_falls_through_unchanged():
    # No file named this way anywhere -- must not raise here (Ultralytics
    # itself is the right place to surface a clear "does not exist"
    # error for a genuinely wrong tracker name).
    bogus = "configs/does_not_exist_anywhere.yaml"
    assert resolve_tracker_config(bogus) == bogus
    print("Part 5: a tracker path that resolves nowhere is returned unchanged, no crash — OK")


def part6_resolve_yolo_weights_unaffected_by_the_new_function():
    # Quick regression guard: adding CONFIGS_DIR/resolve_tracker_config
    # to this module must not have disturbed the pre-existing weights
    # resolution behavior.
    from common.yolo_models import MODELS_DIR
    resolved = resolve_yolo_weights("yolo26n-pose.pt")
    assert resolved == str(MODELS_DIR / "yolo26n-pose.pt"), resolved
    explicit = "/some/other/dir/custom.pt"
    assert resolve_yolo_weights(explicit) == explicit
    print("Part 6: resolve_yolo_weights() still behaves as before — OK")


def main():
    part1_configs_dir_points_at_real_src_configs()
    part2_bare_tracker_name_is_left_untouched()
    part3_configs_relative_path_resolves_regardless_of_cwd()
    part4_already_existing_explicit_path_is_left_untouched()
    part5_unresolvable_path_falls_through_unchanged()
    part6_resolve_yolo_weights_unaffected_by_the_new_function()
    print("\nVerification completed with no errors: tracker/weights path resolution is "
          "independent of the current working directory.")


if __name__ == "__main__":
    main()
