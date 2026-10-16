# tts-overlay

A square, pixel-styled text-to-speech window for Arch Linux + Hyprland, written in Rust
(GTK4). Speech is generated locally with ONNX Runtime and can run on an NVIDIA GPU via CUDA.
Memory use is kept small: the CUDA arena is capped at ~1.5 GB, so it fits cards with 8 GB VRAM or less.

> **Status:** hobby project, only lightly tested (developed on Arch + Hyprland, RTX 5060M).
> Expect rough edges.

## Features

- **Engines:** Kokoro (fast, English; voices can be mixed) and Piper (any voice you drop into a folder, including Ukrainian)
- **UI:** square Material-like look; colors come from your GTK theme
- **Controls:** speed, quality (effective sample rate), pause length (0 = no pause), 8-bit/chiptune effect
- **Cursor tail:** fading squares follow the mouse
- **Visualizer:** pixel audio spectrum in the background
- **Interface language:** English by default, switch to Ukrainian with the flags (voices are chosen separately)
- **Save:** export the last generated audio as a 24 kHz 16-bit mono WAV
- **Hyprland:** a normal tiled window by default; optional layer-shell overlay mode

## Requirements

Arch Linux packages:

```fish
sudo pacman -S --needed rust gtk4 gtk4-layer-shell espeak-ng alsa-lib pipewire-alsa \
    onnxruntime-cuda cuda cudnn
```

`pacman -Ss onnxruntime` shows the exact CUDA package name in your mirror. The app loads the
system `libonnxruntime.so` at runtime (`ORT_DYLIB_PATH`, default `/usr/lib/libonnxruntime.so`).
Without a working CUDA provider it falls back to the CPU (the status line says so).

## Models

Files live in `~/.local/share/tts-overlay/`:

```fish
mkdir -p ~/.local/share/tts-overlay/piper; cd ~/.local/share/tts-overlay

# Kokoro
curl -L https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx -o kokoro.onnx
curl -L https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin -o voices.bin
curl -L https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/config.json -o kokoro-config.json

# Piper (Ukrainian example)
cd piper
set hf https://huggingface.co/rhasspy/piper-voices/resolve/main
curl -LO $hf/uk/uk_UA/ukrainian_tts/medium/uk_UA-ukrainian_tts-medium.onnx
curl -LO $hf/uk/uk_UA/ukrainian_tts/medium/uk_UA-ukrainian_tts-medium.onnx.json
```

`kokoro.onnx` should be roughly 300 MB. If you get `Protobuf parsing failed`, the download is
not a real model (usually an HTML error page); download it again.

Any other Piper voice (`name.onnx` + `name.onnx.json`) placed in `piper/` shows up in the voice list.
Check each model's license before redistributing it.

## Build and run

```fish
cargo build --release
./target/release/tts-overlay
```

Flags: `--hidden` (start without showing the window), `--overlay` (layer-shell overlay instead of a tiled window).

Running the binary again toggles the existing window (single instance). After rebuilding, stop
the old process first: `pkill -f tts-overlay`.

Window class: `dev.local.TtsOverlayRs`. Keys: `Ctrl+Enter` speak, `Esc` hide.

## Notes and limitations

- Kokoro phonemes come from the `espeak-ng` command line, not `misaki`, so pronunciation is somewhat worse than the reference implementation.
- Some Piper voices (e.g. `ukrainian_tts`) use `phoneme_type: text`; the app detects this and skips espeak for them.
- "Quality" only lowers the output sample rate (lo-fi); the models have no real quality setting.
- CUDA on Blackwell (RTX 50xx) depends on how your `onnxruntime-cuda` was built.
- `ort` is pinned to `2.0.0-rc.10` and `gtk4-layer-shell` to `0.4`; bumping them may need small code changes.

## License

MIT, see [LICENSE](LICENSE). Models are covered by their own licenses.
