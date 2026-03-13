# Finger Video Processing Pipeline: Goal, Task, and Attempt Log

## What this project is trying to do
Create a repeatable pipeline that takes a recorded finger-tracking video and its landmark JSON, then produces a clean dataset for downstream modeling/debugging.

## Input data
- Video: `data/finger-video-2026-01-30T22-41-47-949Z.webm`
- Landmarks: `data/finger-landmarks-2026-01-30T22-41-47-949Z.json`

## Required task/output
- Segment each frame so only the human remains, with a white background.
- Create frame exports from the segmented result.
- Create per-frame JSON with index-finger tip position (`x/y`), mapped to frame index.
- Create a debug video from final segmented frames with a visible dot over the index finger position.

## Current implementation
- Script: `data-processing/process_finger_video.py`
- Current outputs:
- `segmented_frames/` (PNG frames)
- `debug_frames/` (PNG frames with red dot overlay)
- `segmented_video.mp4`
- `debug_video.mp4`
- `index_finger_positions.json`

## What we tried so far
1. Initial pipeline build (OpenCV-based)
- Used OpenCV background subtraction (`MOG2`) to separate foreground from background.
- Extracted index-finger tip from MediaPipe-style landmark index `8`.
- Wrote segmented frames, debug overlay frames, and JSON positions.
- Result: pipeline structure worked, but segmentation quality was poor (fragmented/ghosty person mask).

2. Video playback fix
- Original MP4 output had playback issues in your player.
- Switched video encoding to `ffmpeg` with `libx264` and `yuv420p`.
- Result: output videos became playable.

3. Better segmentation model attempt
- Installed `mediapipe` and switched segmentation path to MediaPipe Selfie Segmentation when available.
- Kept MOG2 as fallback if MediaPipe is missing.
- Result: some improvement potential, but still not clean enough in this clip.

4. Mask post-processing improvements
- Added mask smoothing (Gaussian blur), threshold tuning, morphology, and largest-component filtering.
- Added temporal smoothing across frames.
- Added tunable CLI params:
- `--model-selection`
- `--mask-threshold`
- `--mask-blur`
- `--mask-dilate`
- `--temporal-smoothing`
- Ran with aggressive settings (`--mask-threshold 0.10 --mask-blur 17 --mask-dilate 9 --temporal-smoothing 0.8`).
- Result: still missing large portions of the body in segmented output; not acceptable quality yet.

## Current status
- Working: frame export, index-finger per-frame JSON, debug overlay video generation, output video playback.
- Not solved: high-quality human segmentation (clean/full person mask with white background).

## Environment notes observed during work
- Installing `mediapipe` upgraded `numpy` to `2.2.6`.
- Warning observed: installed `pandas` expects `numpy < 2` in this environment.

## Next step to reach target quality
Move from Selfie Segmentation to a stronger person-segmentation model and then keep the same downstream export/debug steps.
