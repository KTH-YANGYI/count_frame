# Reference CSV

This directory stores the bundled reference-frame table:

- `reference_frames.csv`

The current code uses these columns:

- `video_id`
- `reference_time_requested_sec`
- `reference_time_used_sec`
- `reference_frame_idx`
- `roi_rect`
- `roi_source`

They are used to locate the primary reference frame, derive the `reference + 2s` frame, seed `pose_roi`, and derive `workpiece_roi`.

The CSV also contains image-path columns such as `full_frame`, `full_frame_with_roi`, and `crop`. Those paths come from the original manual review workflow and are kept as provenance metadata. The corresponding images are not stored in this Git repository to keep it small.
