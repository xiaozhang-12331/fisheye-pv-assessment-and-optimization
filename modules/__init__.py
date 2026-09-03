# -*- coding: utf-8 -*-
"""Internal functional modules of the refactored pipeline.

The module layout mirrors the execution flow:
1. segmentation: semantic segmentation of the fisheye image.
2. sunpath_occlusion: solar trajectory projection and occlusion detection.
3. radiation: radiation computation.

These modules were migrated from the legacy scripts; future functional
changes should be made here first.
"""
