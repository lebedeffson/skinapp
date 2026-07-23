# Project rules

- The product name is MIRA.
- Keep the Streamlit architecture unless a standalone mobile runtime is explicitly required.
- `MIRA.py` is the only supported launcher on Windows, Linux, and macOS; do not add platform-specific launch scripts.
- Supported desktops are Windows x64, Linux x64, and macOS 14+ on Apple Silicon; fail early on unsupported architectures.
- Dependency versions and CPU-wheel installation live in `MIRA.py`; the generated `.venv` stays ignored.
- Import PyTorch/torchvision before Keras/TensorFlow; the reverse order can crash the shared runtime.
- Keep model imports and constructed models lazy and cached; do not load every mode on the landing screen.
- LAN detection must ignore VPN, container, tunnel, and virtual interfaces.
- Normal launch opens only HTTPS MIRA. The temporary certificate page starts only with `MIRA.py --certificate`.
- Never serve or commit private keys from `.runtime`; only the public CA may be exported.
- Keep Grad-CAM optional because it adds CPU work.
- For details about mobile serving, read `.codex/notes/mobile-local-server.md`.
