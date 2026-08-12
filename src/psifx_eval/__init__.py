"""
psifx_eval
===========
Evaluation framework for psifx's SAM3 cross-chunk ID persistence
(Michele, 2026-08: hired by CHUV to fix psifx's chunk-stitching bugs --
see the module docstrings below and common/../README for the full
context). NOT an independent tracker: this package's job is to run the
REAL psifx (github.com/psifx/psifx) and measure where its cross-chunk
identity linking goes wrong, so a fix can be validated and handed back
to the actual system CHUV runs in production.
"""
