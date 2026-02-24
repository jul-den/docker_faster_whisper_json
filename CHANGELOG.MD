# Changelog

*Русская версия доступна в файле [`CHANGELOG_RU.md`](CHANGELOG_RU.md).*

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.1.0] - 2026-02-24

[1.1.0]: https://github.com/jul-den/docker_faster_whisper_json/releases/tag/v1.1.0

### Added
- Universal `input` format in the `concat` step:
  - Can be a string (single glob pattern) or a list of strings (patterns and/or exact file names).
  - Automatic glob expansion (`*`, `?`, `[]`) for pattern matching.
- New `type: "audio"` for the `concat` step:
  - Concatenation of audio files with automatic normalization to uniform parameters.
  - Parameters `ar` (sample rate) and `ac` (number of channels) can be specified explicitly or set to `"auto"` – the minimum values among all input files are then used.
  - Files with mismatched parameters are transcoded to temporary WAV files, then concatenated using `ffmpeg concat`.
- Helper function `resolve_file_list(input_spec, base_dir)` for unified file lookup.
- Helper function `get_audio_params(file_path)` to retrieve audio parameters via ffprobe.

### Changed
- The `concat` step now handles both video (preserving original copy‑based behavior) and audio.
- Step logging now includes information about transcoding and concatenation.

### Fixed
- Proper handling of cases where input files for `concat` already match the target format (no unnecessary transcoding).

[1.1.0 vs 1.0.0]: https://github.com/jul-den/docker_faster_whisper_json/compare/v1.0.0...v1.1.0

## [1.0.0] - 2026-02-20

[1.0.0]: https://github.com/jul-den/docker_faster_whisper_json/releases/tag/v1.0.0

### Added
- Core functionality: execution of JSON‑defined tasks with the following steps:
  - `concat` – video concatenation (video only, copy‑based using ffmpeg concat demuxer).
  - `extract_audio` – extract audio from video to WAV.
  - `segment` – split audio into segments by timestamps.
  - `transcribe` – speech‑to‑text using Faster Whisper (supports glob patterns and merging results).
  - `llm` – send text to LM Studio (or any OpenAI‑compatible API) with a custom prompt.
- Lazy loading of the Whisper model (initialized only at the first `transcribe_audio` call).
- Skip already completed steps (output file existence check).
- Offline mode via environment variables `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE`.
- Per‑task override of global settings (Whisper model, LLM model, URL, token, WAV deletion) via the `settings` object in each JSON.
- Batch processing of multiple JSON files placed in `/source`.
- Logging with `flush=True` for proper output in Docker logs.
- Basic documentation in `README_RU.md` and `README.md`.

### Known limitations
- The `concat` step supported only video (copy‑based concatenation). Audio had to be prepared manually or processed separately.
- Input specification for `concat` was limited to a single glob string (no list support).
- Concatenating audio required all files to be pre‑normalized to the same format.
