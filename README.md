# Segmentation Images — README

## Purpose
Small utility/data-generation codebase for creating synthetic/annotated PNG segmentation images of surgical items (scissors, lap sponges, gauze, empty frames) used for model training and data augmentation.


## How it works (summary)
1. Load a background.
2. Sample number and types of items according to configured probabilities.
3. For each sampled item: load asset, apply resizing/resampling/augmentation, place on background, update segmentation mask.
4. Save RGB PNG and per-pixel class masks (PNG or indexed format).

## Current issues (noted)
1. Some images (scissors, lap sponge, gauze) are very large — resizing isn't working as intended and may be too randomized.
2. Item rarity: approximately 50% of generated frames should be empty. Scissors should be exceedingly rare; sponges and gauze should be more common.
3. Plan requested to convert output from PNGs to DICOM. This is feasible but a larger undertaking (metadata, pixel format, storage, viewer compatibility).