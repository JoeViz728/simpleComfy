**Note.** The code in this project was generated automatically to match my design, and I tested it from the user side.

# simpleComfy

A local front end for ComfyUI. It is meant to use ComfyUI in a simple, user-friendly way, without working in the graph editor.

You pick a preset, write a prompt, and the app builds the graph, queues the job, and pulls missing weights before the job runs. One GPU job runs at a time. The queue keeps its order across restarts, and a running job can be cancelled.

Presets, effects, and video packs are YAML files. The files in this repo are empty samples. Point them at weights you have a license to use.

## To do

- Refine the workflows.
- Add a larger viewer for the output image.
- Add modes for video voiceover and music, image editing with marked focus areas, proper upscaling, character consistency, text to speech, and 3D.
- Make models modular, so a setup is a choice rather than a fixed pack.
- Finish the API.
- Add Windows support.

## Run

```bash
cp config.example.toml config.toml
./scripts/install.sh
./scripts/start.sh
```

The app listens on `127.0.0.1:8790`. ComfyUI has to be running, or named in the config so the app can start it. This copy is Linux only.

```bash
pytest
```
