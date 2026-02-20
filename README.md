# Universal JSON-oriented video/audio processor based on Faster Whisper

*Русская версия доступна в файле [`README_RU.md`](README_RU.md).*

This tool automates the transcription of webinars, lectures, and other video/audio recordings, followed by text processing with an LLM (e.g., via LM Studio). Everything is controlled through JSON files that define a sequence of steps: concatenating fragments, extracting audio, cutting by timestamps, transcribing with Faster Whisper, and sending requests to an LLM.

## Features

- **Batch processing** – place several JSON tasks in the `/source` folder; the script will execute them one by one.
- **Flexible step definition** – supported operations:
  - `concat` – concatenate video files (e.g., multiple `.ts` into one `.mp4`).
  - `extract_audio` – extract audio track from video to WAV (16 kHz, mono).
  - `segment` – split audio into segments by timestamps.
  - `transcribe` – speech recognition (Whisper) with glob patterns for multiple files.
  - `llm` – send text to an LLM (LM Studio) with a custom prompt.
- **Lazy loading of Whisper model** – the model is loaded only at the first `transcribe_audio` call.
- **Skip already completed steps** – if an output file already exists, the step is not re-executed.
- **Offline mode** – after the model has been downloaded once, all subsequent operations work without offline.
- **Override settings** – each JSON can define its own parameters (Whisper model, LLM model, URL, etc.) which take precedence over environment variables.

## Requirements

- **Docker** installed (preferably with volume mounting support).
- **LM Studio** (optional, only if you use `llm` steps). The server must be running and accessible from the container (typically at `http://host.docker.internal:1234` for Windows/macOS).
- **Whisper models** are cached locally (usually in ~/.cache/huggingface/hub/). On the first run without cache, the model will be downloaded – this is normal, but for offline mode the model will remain in the cache after the first download.

## JSON Task Structure

Each JSON file must contain the following fields:

- `type` (required) – must be `"video_transcription"` (the script ignores files with other types).
- `name` (optional) – task name; used to create a subfolder in `/output`. If omitted, the file name (without extension) is used.
- `settings` (optional) – an object with parameters overriding global environment variables:
  - `whisper_model` – Whisper model (e.g., `"large-v3"`, `"medium"`).
  - `lm_model` – model used in LM Studio (e.g., `"qwen/qwen3-8b"`).
  - `lm_url` – LM Studio endpoint URL (default `http://host.docker.internal:1234/v1/chat/completions`).
  - `lm_token` – API access token (default none).
  - `delete_wav` – whether to delete temporary `.wav` files after transcription (`true`/`false`).
- `steps` – an array of steps. Each step contains an `action` field and additional parameters depending on the action.

### Step Parameters

#### `concat` – concatenate files (any files compatible with FFmpeg concat demuxer)
- `input` – file glob pattern (e.g., `"media_*.ts"`).
- `output` – name of the resulting mp4 file.
- `type` – `"video"` (only video is supported).

#### `extract_audio` – extract audio
- `input` – input video file name.
- `output` – output WAV file name.
- `params` – object with ffmpeg parameters (default `{"ar": 16000, "ac": 1}`).

#### `segment` – split audio into segments
- `input` – source audio file name.
- `output_prefix` – prefix for segment file names (a number and extension will be appended).
- `segments` – array of objects with `start` and `end` fields (in seconds).
- `format` – output file extension (default `"wav"`).

#### `transcribe` – transcribe audio
- `input` – audio file glob pattern (e.g., `"part2_doklad_seg*.wav"`). Files are searched first in the task folder, then in `/source`.
- `output` – name of the text file where the result will be written.
- `language` – language (default `"ru"`).
- `beam_size` – Whisper parameter (default `5`).
- `combine` – if `true` (default), texts from all matched files are joined with a newline character. If `false`, only the first file (alphabetically) is used.

#### `llm` – request to LM Studio
- `input` – name of the text file containing the source text.
- `output` – file name for the LLM response.
- `prompt` – prompt text. Use the placeholder `{input_text}` which will be replaced by the content of the input file.
- `system` – optional system prompt.
- `max_tokens` – maximum number of tokens in the response (default `8000`).

### Example JSON for a Webinar (Multiple Steps)

```json
{
  "type": "video_transcription",
  "name": "webinar",
  "settings": {
    "delete_wav": true,
    "whisper_model": "large-v3",
    "lm_model": "qwen/qwen3-8b"
  },
  "steps": [
    {
      "action": "concat",
      "input": "media_*.ts",
      "output": "full_video.mp4",
      "type": "video"
    },
    {
      "action": "extract_audio",
      "input": "full_video.mp4",
      "output": "full_audio.wav"
    },
    {
      "action": "segment",
      "input": "full_audio.wav",
      "output_prefix": "part2_doklad_seg",
      "segments": [
        {"start": 252.0, "end": 1440.0}
      ]
    },
    {
      "action": "transcribe",
      "input": "part2_doklad_seg*.wav",
      "output": "part2_doklad.txt",
      "language": "ru"
    },
    {
      "action": "llm",
      "input": "part2_doklad.txt",
      "output": "summary_doklad.txt",
      "prompt": "Summarize the presentation based on the following transcript:\n\n{input_text}"
    },
    {
      "action": "segment",
      "input": "full_audio.wav",
      "output_prefix": "part4_qa_seg",
      "segments": [
        {"start": 1485.0, "end": 4419.0}
      ]
    },
    {
      "action": "transcribe",
      "input": "part4_qa_seg*.wav",
      "output": "part4_qa.txt",
      "language": "ru"
    },
    {
      "action": "llm",
      "input": "part4_qa.txt",
      "output": "qa_pairs.txt",
      "prompt": "Below is the transcript of a Q&A session from a webinar. Extract all question-answer pairs. Ignore the host's words like \"Next question\", \"We have a question\", and technical remarks. Output each pair in the format:\n\nQuestion: <question text>\nAnswer: <answer text>\n---\n(separator between pairs)\n\nTranscript:\n{input_text}",
      "max_tokens": 8000
    },
    {
      "action": "transcribe",
      "input": "full_audio.wav",
      "output": "full_transcript.txt",
      "language": "ru"
    }
  ]
}
```

## Running the Docker Container

1. **Build the image** (if not already done):
   ```bash
   docker build -t webinar-processor .
   ```

2. **Prepare folders**:
   - Create a folder on your host machine for source files. Place all required files (e.g., `.ts` files) and the JSON task there.
   - Create a folder for the results.

3. **Start the container** (example for Windows PowerShell):
   ```powershell
   docker run -d `
     --name webinar-processor-container `
     -v DRIVE:\PATH\TO\source:/source `
     -v DRIVE:\PATH\TO\output:/app/output `
     -e VIDEO_DIR=/source `
     -e OUTPUT_DIR=/app/output `
     -e LLM_URL="http://host.docker.internal:1234/v1/chat/completions" `
     -e LLM_API_TOKEN="TOKEN" `
     webinar-processor
   ```
   On Linux, replace `host.docker.internal` with `localhost` and add `--network="host"`.

4. **View logs**:
   ```bash
   docker logs -f webinar-processor-container
   ```

5. **Stop and remove**:
   ```bash
   docker stop webinar-processor-container
   docker rm webinar-processor-container
   ```

## Environment Variables (can be set at runtime, but not mandatory)

- `VIDEO_DIR` – folder with source files and JSON tasks (default `/source`).
- `OUTPUT_DIR` – folder for results (default `/app/output`).
- `LLM_URL` – LLM endpoint URL (LM Studio, default `http://host.docker.internal:1234/v1/chat/completions`). You can also specify any OpenAI‑compatible LLM endpoint.
- `LLM_API_TOKEN` – API access token (default not used).
- `LLM_MODEL` – default model for LLM steps (default `qwen/qwen3-8b`).
- `WHISPER_MODEL` – Whisper model (default `large-v3`).
- `DELETE_WAV_AFTER_TRANSCRIBE` – whether to delete temporary WAV files after transcription (`true`/`false`). Default `true`.

## Notes

- The Whisper model is loaded **lazily** – only at the first call to `transcribe_audio`. If all texts are already transcribed (the `.txt` files exist), the model will not be loaded at all.
- For offline mode, it is recommended to pre‑cache the required Whisper model (e.g., by running a one‑time transcription with internet access). After that you can set the environment variables `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` (they are already set inside `get_whisper_model`).
- All paths in JSON are interpreted as relative:
  - input files are first looked up in `/output/<task_name>/`, then in `/source/`;
  - output files are always written to `/output/<task_name>/`.
- If a glob pattern is used in a `transcribe` step and `combine: true`, all matching texts are joined. Intermediate text files for each segment are saved with the same base name but with a `.txt` extension (next to the WAV files).
- If any step fails, the execution of the current task is stopped (the script moves to the next JSON file).

## Support

If you have any questions or suggestions, please open an issue or contact the author.
