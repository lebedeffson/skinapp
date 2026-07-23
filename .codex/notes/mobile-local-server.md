# Mobile local-server notes

The low-change mobile path is to run Streamlit on the computer and open it from
a phone on the same Wi-Fi network. The ML runtime and model files remain on the
computer.

`MIRA.py` is the only launcher. It creates `.venv`, installs pinned dependencies,
generates HTTPS material, binds Streamlit to `0.0.0.0`, and prints a LAN address.
Do not derive that address from the default route: a full-tunnel VPN may return
an address the phone cannot reach. Prefer an active physical interface and
exclude tunnel, VPN, container, and virtual bridge interfaces.

The current Python binary stack must import PyTorch and torchvision before
Keras/TensorFlow. Importing Keras/TensorFlow first can crash while PyTorch loads
Triton. Keep the launcher dependency probe in the same safe order.

`st.camera_input` requires a trusted secure context on a phone. `MIRA.py` creates
a local CA and IP-address certificate under `.runtime`. Normal launch opens only
the HTTPS application and does not listen on the helper HTTP port. For a new
phone, `MIRA.py --certificate` temporarily serves only the public CA. Never
expose, commit, or copy private CA/server keys.

Keep `st.file_uploader` as a fallback: mobile operating systems normally offer
both camera capture and gallery selection from the file chooser.

Streamlit tabs execute all tab bodies, so they caused every model to load during
the first render. The landing view now uses a segmented control, and model
imports plus constructed PyTorch models are cached lazily. Preserve this layout
unless there is a measured reason to change it.

The tested dependency set is embedded in `MIRA.py`: Python 3.12–3.13,
Streamlit 1.60, TensorFlow 2.21, Keras 3.15, PyTorch 2.13, and torchvision 0.28.
Windows and Linux use the official CPU PyTorch index; macOS uses the standard
wheel. Always probe imports in PyTorch-first order after installation.
The current cross-platform target is Windows x64, Linux x64, and macOS 14+ on
Apple Silicon. Fail before installation on Intel Mac or unsupported
architectures so users do not download several gigabytes before a wheel error.

Grad-CAM is optional and disabled by default so the primary prediction appears
quickly. Keras Sequential models need layer-by-layer forward execution inside
the gradient tape; the ordinary functional Grad-CAM graph can return no
gradient for the restored emotion model.

A standalone APK is not a packaging-only change for this project. It would
require an Android-compatible inference stack or a separate inference server.
