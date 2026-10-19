**Note.** The code in this project was generated automatically to match my design, and I tested it from the user side.

# simpleComfy

A local image and video front end for ComfyUI. Python, FastAPI, and a small browser UI.

You pick a preset, write a prompt, and the service builds the ComfyUI graph, queues the job, and pulls any missing weights before it runs. One GPU job runs at a time. The queue keeps its order across restarts, and a running job can be cancelled.

Presets, effects, and video packs are YAML files. The files in this repo are empty samples, so the app runs and the tests pass without pinning anyone's model choices. Point them at weights you have a license to use.

## Run

```bash
cp config.example.toml config.toml
./scripts/install.sh
./scripts/start.sh
```

The service listens on `127.0.0.1:8790`. ComfyUI has to be running, or named in the config so the service can start it.

```bash
pytest
```
