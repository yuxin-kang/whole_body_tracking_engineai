# Interactive NPZ Trim Design

## Goal

Add an independent Isaac Sim replay tool that lets the operator pause playback, mark a logical start frame and end frame, and export a trimmed `.npz` motion file without changing the input file format.

## Requirements

- The feature lives in a new script: `scripts/trim_npz.py`.
- Playback uses logical motion frames, not wall-clock time.
- The operator uses keyboard shortcuts in the Isaac Sim window.
- Trimming uses a closed interval `[start_frame, end_frame]`.
- The output keeps the original `.npz` key set, field names, dtypes, and metadata semantics.
- Arrays whose leading dimension matches the source frame count are trimmed on the time axis.
- Non-time metadata such as `fps` is preserved unchanged.
- If `--output_file` is omitted, the script auto-generates `*_trim_<start>_<end>.npz`.

## Interaction Model

The session tracks:

- `current_frame`
- `is_playing`
- `start_frame`
- `end_frame`
- `pending_export`

Keyboard shortcuts:

- `Space`: toggle play/pause
- `Left`: step backward one frame while paused
- `Right`: step forward one frame while paused
- `[` or `S`: set start marker at current frame
- `]` or `E`: set end marker at current frame
- `C`: clear both markers
- `Enter`: export when both markers are set and ordered
- `Q` or `Esc`: exit

The terminal prints the current frame, logical time, playback state, markers, and export path hints.

## Implementation Shape

Split the work into two layers:

1. Pure trim helpers in `source/whole_body_tracking/whole_body_tracking/utils/motion_trim.py`
2. Isaac Sim interactive script in `scripts/trim_npz.py`

The helper layer owns:

- loading `.npz` contents into numpy arrays
- validating frame-aligned keys
- deriving output paths
- trimming arrays on the leading time dimension
- writing the output archive

The script layer owns:

- Isaac Sim setup
- motion replay
- keyboard event subscription
- marker management
- calling helper functions for export

## Error Handling

- Reject missing or non-`.npz` inputs.
- Reject invalid marker order instead of silently swapping.
- Reject overwrite unless `--force` is set.
- Reject inconsistent time-series leading dimensions.
- Keep runtime failures explicit in terminal output.

## Testing Strategy

Use test-first coverage for the pure helper layer:

- trim closed interval behavior
- preservation of non-time metadata
- rejection of inconsistent time-series shapes
- auto-generated output name behavior
- overwrite protection behavior

The interactive Isaac Sim path is validated with syntax checks and targeted source inspection because the test environment does not provide a live Omniverse window.
