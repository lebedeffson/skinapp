#!/usr/bin/env python3
"""One-file bootstrap and launcher for MIRA on Windows, Linux, and macOS."""
from __future__ import print_function

import argparse
import contextlib
import hashlib
import http.server
import ipaddress
import json
import os
import platform
import shutil
import socket
import ssl
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
VENV_DIR = APP_DIR / ".venv"
RUNTIME_DIR = APP_DIR / ".runtime"
PUBLIC_DIR = RUNTIME_DIR / "public"
DEPENDENCY_MARKER = VENV_DIR / ".mira-dependencies.json"

PYTHON_MIN = (3, 12)
PYTHON_MAX = (3, 13)

GENERAL_PACKAGES = (
    "streamlit==1.60.0",
    "tensorflow==2.21.0",
    "keras==3.15.0",
    "numpy==2.5.1",
    "pandas==2.3.3",
    "matplotlib==3.11.0",
    "Pillow==12.3.0",
    "cryptography==49.0.0",
    "psutil==7.2.2",
)
TORCH_PACKAGES = ("torch==2.13.0", "torchvision==0.28.0")
TORCH_CPU_PACKAGES = ("torch==2.13.0+cpu", "torchvision==0.28.0+cpu")

MODEL_FILES = (
    "skin_disease_model_focal_recall_FINAL.keras",
    "final_fine_tuned_medical_model (2).pth",
    "final_nails_model (1).pth",
    "emotion_model_attempt_63_plus.keras",
)


def log(message=""):
    print(message, flush=True)


def fail(message, code=1):
    log("")
    log("ОШИБКА: " + message)
    if os.name == "nt" and sys.stdin.isatty():
        with contextlib.suppress(EOFError):
            input("Нажмите Enter, чтобы закрыть окно...")
    raise SystemExit(code)


def supported_python(version):
    return version[0] == 3 and PYTHON_MIN[1] <= version[1] <= PYTHON_MAX[1]


def validate_platform():
    machine = platform.machine().lower()
    if sys.platform == "win32":
        if machine not in ("amd64", "x86_64"):
            fail("На Windows нужен 64-битный компьютер x64.")
        return

    if sys.platform.startswith("linux"):
        if machine not in ("amd64", "x86_64"):
            fail("На Linux сейчас поддерживаются 64-битные компьютеры x64.")
        return

    if sys.platform == "darwin":
        if machine not in ("arm64", "aarch64"):
            fail(
                "На macOS нужен Mac с Apple Silicon (M1 или новее) и нативный "
                "ARM64 Python. Для Intel Mac актуальные ML-библиотеки не "
                "выпускают совместимые пакеты."
            )
        version_text = platform.mac_ver()[0]
        try:
            major = int(version_text.split(".", 1)[0])
        except (TypeError, ValueError):
            major = 0
        if major and major < 14:
            fail("Для MIRA на Mac требуется macOS 14 или новее.")
        return

    fail("Поддерживаются Windows x64, Linux x64 и macOS на Apple Silicon.")


def python_version(command):
    try:
        result = subprocess.run(
            list(command)
            + [
                "-c",
                "import json,struct,sys;"
                "print(json.dumps([sys.version_info[0],sys.version_info[1],"
                "struct.calcsize('P')*8,sys.executable]))",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        major, minor, bits, executable = json.loads(result.stdout.strip())
        return (major, minor), bits, executable
    except (OSError, ValueError, subprocess.SubprocessError, json.JSONDecodeError):
        return None


def candidate_pythons():
    candidates = []
    configured = os.environ.get("MIRA_PYTHON")
    if configured:
        candidates.append([configured])

    candidates.append([sys.executable])
    if os.name == "nt":
        candidates.extend((["py", "-3.12"], ["py", "-3.13"], ["python"]))
    else:
        candidates.extend(
            (
                ["python3.12"],
                ["python3.13"],
                ["python3"],
            )
        )

    uv = shutil.which("uv")
    if uv:
        for version in ("3.12", "3.13"):
            try:
                result = subprocess.run(
                    [uv, "python", "find", version],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                path = result.stdout.strip()
                if path:
                    candidates.append([path])
            except (OSError, subprocess.SubprocessError):
                continue

    unique = []
    seen = set()
    for command in candidates:
        key = tuple(command)
        if key not in seen:
            unique.append(command)
            seen.add(key)
    return unique


def choose_bootstrap_python():
    for command in candidate_pythons():
        details = python_version(command)
        if not details:
            continue
        version, bits, executable = details
        if supported_python(version) and bits == 64:
            return command, version, executable

    uv = shutil.which("uv")
    if uv:
        log("Совместимый Python не найден. Устанавливаю Python 3.12 через uv...")
        try:
            subprocess.run([uv, "python", "install", "3.12"], check=True)
            result = subprocess.run(
                [uv, "python", "find", "3.12"],
                check=True,
                capture_output=True,
                text=True,
            )
            command = [result.stdout.strip()]
            details = python_version(command)
            if details and supported_python(details[0]) and details[1] == 64:
                return command, details[0], details[2]
        except (OSError, subprocess.SubprocessError):
            pass

    fail(
        "Нужен 64-битный Python 3.12 или 3.13. Установите Python 3.12 с "
        "https://www.python.org/downloads/ и снова запустите этот файл."
    )


def venv_python():
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def dependency_signature():
    payload = {
        "general": GENERAL_PACKAGES,
        "torch": TORCH_PACKAGES,
        "torch_cpu": TORCH_CPU_PACKAGES,
        "platform": sys.platform,
        "machine": platform.machine().lower(),
    }
    serialized = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def create_venv(python_command):
    log("Создаю изолированное окружение .venv...")
    subprocess.run(
        list(python_command) + ["-m", "venv", str(VENV_DIR)],
        cwd=str(APP_DIR),
        check=True,
    )


def fast_dependency_probe(python):
    probe = (
        "import importlib.util,sys;"
        "names=('streamlit','torch','torchvision','tensorflow','keras',"
        "'numpy','pandas','matplotlib','PIL','cryptography','psutil');"
        "sys.exit(0 if all(importlib.util.find_spec(n) for n in names) else 1)"
    )
    return (
        subprocess.run(
            [str(python), "-c", probe],
            cwd=str(APP_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def dependencies_current(python):
    if not DEPENDENCY_MARKER.exists() or not fast_dependency_probe(python):
        return False
    try:
        data = json.loads(DEPENDENCY_MARKER.read_text(encoding="utf-8"))
        return data.get("signature") == dependency_signature()
    except (OSError, ValueError):
        return False


def install_dependencies(python, reinstall=False):
    if not reinstall and dependencies_current(python):
        log("Зависимости уже установлены.")
        return

    log("")
    log("Устанавливаю зависимости. Первый запуск может занять 5–15 минут.")
    log("Повторные запуски ничего скачивать не будут.")
    subprocess.run(
        [str(python), "-m", "pip", "install", "--upgrade", "pip", "wheel"],
        cwd=str(APP_DIR),
        check=True,
    )

    common_command = [
        str(python),
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--only-binary=:all:",
    ]
    if reinstall:
        common_command.append("--force-reinstall")
    subprocess.run(
        common_command + list(GENERAL_PACKAGES),
        cwd=str(APP_DIR),
        check=True,
    )

    if sys.platform == "darwin":
        torch_command = common_command + list(TORCH_PACKAGES)
    else:
        torch_command = (
            common_command
            + [
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
            ]
            + list(TORCH_CPU_PACKAGES)
        )
    subprocess.run(torch_command, cwd=str(APP_DIR), check=True)

    log("Проверяю совместный импорт PyTorch и TensorFlow...")
    subprocess.run(
        [
            str(python),
            "-X",
            "faulthandler",
            "-c",
            "import streamlit; import torch; import torchvision; "
            "import keras; import tensorflow; import cryptography; import psutil; "
            "print('ML runtime: OK')",
        ],
        cwd=str(APP_DIR),
        check=True,
    )
    DEPENDENCY_MARKER.write_text(
        json.dumps(
            {
                "signature": dependency_signature(),
                "created_at": datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def ensure_environment(reinstall=False):
    command, version, executable = choose_bootstrap_python()
    log(
        "Python: {0} ({1}.{2}, 64-bit)".format(
            executable,
            version[0],
            version[1],
        )
    )

    python = venv_python()
    valid_venv = False
    if python.exists():
        details = python_version([str(python)])
        valid_venv = bool(
            details and supported_python(details[0]) and details[1] == 64
        )
    if not valid_venv:
        if VENV_DIR.exists():
            log("Старое окружение .venv несовместимо. Пересоздаю его...")
            if VENV_DIR.is_symlink():
                VENV_DIR.unlink()
            else:
                shutil.rmtree(VENV_DIR)
        create_venv(command)
    install_dependencies(python, reinstall=reinstall)
    return python


def validate_project_files():
    missing = [
        name for name in ("main.py",) + MODEL_FILES if not (APP_DIR / name).is_file()
    ]
    if missing:
        fail("Не найдены файлы: " + ", ".join(missing))


def detect_lan_ip():
    import psutil

    ignored = (
        "lo",
        "loopback",
        "tun",
        "tap",
        "vpn",
        "wg",
        "tailscale",
        "docker",
        "veth",
        "virbr",
        "vmnet",
        "hyper-v",
        "virtualbox",
        "bridge",
    )
    preferred = ("wi-fi", "wifi", "wlan", "wireless", "ethernet", "eth", "en0")
    candidates = []
    stats = psutil.net_if_stats()

    for interface, addresses in psutil.net_if_addrs().items():
        name = interface.lower()
        if any(token in name for token in ignored):
            continue
        if interface in stats and not stats[interface].isup:
            continue
        for address in addresses:
            if address.family != socket.AF_INET:
                continue
            try:
                parsed = ipaddress.ip_address(address.address)
            except ValueError:
                continue
            if not parsed.is_private or parsed.is_loopback or parsed.is_link_local:
                continue
            score = 0
            if any(token in name for token in preferred):
                score += 100
            if address.address.startswith("192.168."):
                score += 30
            elif address.address.startswith("10."):
                score += 20
            elif address.address.startswith("172."):
                score += 10
            candidates.append((score, interface, address.address))

    if candidates:
        return sorted(candidates, reverse=True)[0][2]

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        fail("Не удалось определить IP Wi-Fi/Ethernet.")


def write_private(path, data):
    path.write_bytes(data)
    if os.name != "nt":
        path.chmod(0o600)


def certificate_expiry(cert):
    value = getattr(cert, "not_valid_after_utc", None)
    if value is not None:
        return value
    return cert.not_valid_after.replace(tzinfo=timezone.utc)


def ensure_certificates(lan_ip, app_port):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    ca_key_path = RUNTIME_DIR / "mira-local-ca.key"
    ca_cert_path = RUNTIME_DIR / "mira-local-ca.crt"
    server_key_path = RUNTIME_DIR / "mira-server.key"
    server_cert_path = RUNTIME_DIR / "mira-server.crt"

    legacy_dir = APP_DIR / ".certs"
    legacy_files = {
        ca_key_path: legacy_dir / "skinapp-local-ca.key",
        ca_cert_path: legacy_dir / "skinapp-local-ca.crt",
        server_key_path: legacy_dir / "skinapp-server.key",
        server_cert_path: legacy_dir / "skinapp-server.crt",
    }
    for destination, source in legacy_files.items():
        if not destination.exists() and source.exists():
            shutil.copy2(str(source), str(destination))

    now = datetime.now(timezone.utc)
    created_ca = False
    if not ca_key_path.exists() or not ca_cert_path.exists():
        ca_key = rsa.generate_private_key(public_exponent=65537, key_size=3072)
        subject = x509.Name(
            [x509.NameAttribute(NameOID.COMMON_NAME, "MIRA Local CA")]
        )
        ca_cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(subject)
            .public_key(ca_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(days=1))
            .not_valid_after(now + timedelta(days=3650))
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .add_extension(
                x509.KeyUsage(
                    digital_signature=False,
                    content_commitment=False,
                    key_encipherment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    key_cert_sign=True,
                    crl_sign=True,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .add_extension(
                x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()),
                critical=False,
            )
            .sign(ca_key, hashes.SHA256())
        )
        write_private(
            ca_key_path,
            ca_key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            ),
        )
        ca_cert_path.write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))
        created_ca = True
    else:
        ca_key = serialization.load_pem_private_key(
            ca_key_path.read_bytes(),
            password=None,
        )
        ca_cert = x509.load_pem_x509_certificate(ca_cert_path.read_bytes())

    needs_server_cert = True
    if server_key_path.exists() and server_cert_path.exists():
        try:
            existing = x509.load_pem_x509_certificate(server_cert_path.read_bytes())
            sans = existing.extensions.get_extension_for_class(
                x509.SubjectAlternativeName
            ).value
            addresses = {
                str(value)
                for value in sans.get_values_for_type(x509.IPAddress)
            }
            needs_server_cert = (
                lan_ip not in addresses
                or certificate_expiry(existing) < now + timedelta(days=1)
                or existing.issuer != ca_cert.subject
            )
        except (ValueError, x509.ExtensionNotFound):
            needs_server_cert = True

    if needs_server_cert:
        server_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, lan_ip)])
        server_cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(ca_cert.subject)
            .public_key(server_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(days=1))
            .not_valid_after(now + timedelta(days=397))
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), True)
            .add_extension(
                x509.SubjectAlternativeName(
                    [
                        x509.IPAddress(ipaddress.ip_address(lan_ip)),
                        x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                        x509.DNSName("localhost"),
                    ]
                ),
                critical=False,
            )
            .add_extension(
                x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),
                critical=False,
            )
            .sign(ca_key, hashes.SHA256())
        )
        write_private(
            server_key_path,
            server_key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            ),
        )
        server_cert_path.write_bytes(
            server_cert.public_bytes(serialization.Encoding.PEM)
        )

    public_ca = PUBLIC_DIR / "mira-local-ca.crt"
    public_ca.write_bytes(ca_cert_path.read_bytes())
    fingerprint = ca_cert.fingerprint(hashes.SHA256()).hex().upper()
    grouped_fingerprint = ":".join(
        fingerprint[index : index + 2] for index in range(0, len(fingerprint), 2)
    )
    (PUBLIC_DIR / "index.html").write_text(
        certificate_page(lan_ip, app_port, grouped_fingerprint),
        encoding="utf-8",
    )
    return ca_cert_path, server_cert_path, server_key_path, created_ca


def certificate_page(lan_ip, app_port, fingerprint):
    return """<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MIRA — сертификат</title>
  <style>
    body {{ font: 17px/1.5 system-ui,sans-serif; max-width: 720px; margin: auto; padding: 24px; color: #172033; }}
    a.button {{ display: inline-block; padding: 14px 18px; border-radius: 12px; background: #1769e0; color: #fff; text-decoration: none; font-weight: 700; }}
    code {{ overflow-wrap: anywhere; }}
    li {{ margin: 12px 0; }}
  </style>
</head>
<body>
  <h1>MIRA: включение камеры</h1>
  <ol>
    <li><a class="button" href="/mira-local-ca.crt">Скачать сертификат</a></li>
    <li><b>Android:</b> Настройки → Безопасность → Установить сертификат → Сертификат ЦС.</li>
    <li><b>iPhone/iPad:</b> установите профиль, затем Настройки → Основные → Об этом устройстве → Доверие сертификатам.</li>
    <li>Откройте <a href="https://{ip}:{port}">https://{ip}:{port}</a> и разрешите камеру.</li>
  </ol>
  <p>SHA-256: <code>{fingerprint}</code></p>
  <p>Закрытые ключи остаются на компьютере. Здесь опубликован только сертификат локального центра.</p>
</body>
</html>
""".format(ip=lan_ip, port=app_port, fingerprint=fingerprint)


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".crt": "application/x-x509-ca-cert",
    }

    def log_message(self, fmt, *args):
        return


def port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", port))
            return True
        except OSError:
            return False


def start_ca_server(port):
    if not port_available(port):
        fail(
            "Порт страницы сертификата {0} занят. "
            "Укажите другой: --ca-port 8510".format(port)
        )
    handler = partial(QuietHandler, directory=str(PUBLIC_DIR))
    server = http.server.ThreadingHTTPServer(("0.0.0.0", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def wait_for_https(url, ca_cert, process, timeout=30):
    context = ssl.create_default_context(cafile=str(ca_cert))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, context=context, timeout=2) as response:
                if response.status == 200:
                    return True
        except (OSError, urllib.error.URLError):
            time.sleep(0.25)
    return False


def runtime_doctor():
    import cryptography
    import psutil
    import streamlit
    import torch
    import torchvision
    import keras
    import tensorflow

    log("Streamlit:   " + streamlit.__version__)
    log("PyTorch:     " + torch.__version__)
    log("torchvision: " + torchvision.__version__)
    log("TensorFlow:  " + tensorflow.__version__)
    log("Keras:       " + keras.__version__)
    log("cryptography:" + cryptography.__version__)
    log("psutil:      " + psutil.__version__)
    log("Модели:      OK")


def run_runtime(args):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    validate_project_files()
    lan_ip = detect_lan_ip()
    ca_cert, server_cert, server_key, created_ca = ensure_certificates(
        lan_ip,
        args.port,
    )

    if args.doctor:
        runtime_doctor()
        log("LAN IP:      " + lan_ip)
        log("HTTPS cert:  OK")
        return 0

    if args.certificate:
        ca_server = start_ca_server(args.ca_port)
        ca_url = "http://{0}:{1}".format(lan_ip, args.ca_port)
        log("Сертификат MIRA: " + ca_url)
        log("После установки сертификата нажмите Ctrl+C.")
        if not args.no_browser:
            webbrowser.open(ca_url)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            return 0
        finally:
            ca_server.shutdown()
            ca_server.server_close()

    if not port_available(args.port):
        fail(
            "Порт приложения {0} занят. Остановите старый сервер или используйте "
            "--port 8502.".format(args.port)
        )

    app_url = "https://{0}:{1}".format(lan_ip, args.port)
    local_url = "https://localhost:{0}".format(args.port)

    log("")
    log("=" * 56)
    log("MIRA")
    log("=" * 56)
    log("На компьютере: " + local_url)
    log("На телефоне:   " + app_url)
    log("")
    log("Телефон и компьютер должны быть в одной Wi-Fi сети.")
    if created_ca:
        log(
            "Создан новый HTTPS-сертификат. Для нового телефона один раз "
            "запустите: python MIRA.py --certificate"
        )
    log("Для остановки нажмите Ctrl+C.")
    log("")

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_DIR / "main.py"),
        "--server.address",
        "0.0.0.0",
        "--server.port",
        str(args.port),
        "--server.fileWatcherType",
        "none",
        "--server.enableWebsocketCompression",
        "true",
        "--server.sslCertFile",
        str(server_cert),
        "--server.sslKeyFile",
        str(server_key),
        "--browser.gatherUsageStats",
        "false",
    ]
    environment = os.environ.copy()
    environment.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    environment.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    process = subprocess.Popen(command, cwd=str(APP_DIR), env=environment)
    health_url = app_url + "/_stcore/health"

    try:
        if not wait_for_https(health_url, ca_cert, process):
            process.terminate()
            fail("HTTPS-сервер не прошёл проверку запуска.")
        log("Сервер готов.")
        if not args.no_browser:
            webbrowser.open(app_url)
        return process.wait()
    except KeyboardInterrupt:
        log("\nОстанавливаю MIRA...")
        process.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=10)
        if process.poll() is None:
            process.kill()
        return 0


def parse_args(argv=None):
    def port_number(value):
        try:
            port = int(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError("порт должен быть числом") from error
        if not 1 <= port <= 65535:
            raise argparse.ArgumentTypeError("порт должен быть от 1 до 65535")
        return port

    parser = argparse.ArgumentParser(
        description="Установка зависимостей и запуск MIRA",
    )
    parser.add_argument(
        "--port",
        type=port_number,
        default=8501,
        help="HTTPS-порт приложения",
    )
    parser.add_argument(
        "--ca-port",
        type=port_number,
        default=8500,
        help="HTTP-порт страницы установки сертификата",
    )
    parser.add_argument(
        "--certificate",
        action="store_true",
        help="открыть временную страницу установки сертификата для нового телефона",
    )
    parser.add_argument(
        "--reinstall",
        action="store_true",
        help="переустановить все зависимости",
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="проверить окружение и модели без запуска сервера",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="не открывать браузер автоматически",
    )
    parser.add_argument("--runtime", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    os.chdir(APP_DIR)
    validate_platform()
    if args.runtime:
        return run_runtime(args)

    validate_project_files()
    try:
        python = ensure_environment(reinstall=args.reinstall)
    except subprocess.CalledProcessError as error:
        fail(
            "Не удалось подготовить библиотеки MIRA. Проверьте интернет и "
            "свободное место, затем повторите запуск.",
            code=error.returncode or 1,
        )
    except OSError as error:
        fail("Не удалось подготовить окружение: {0}".format(error))
    command = [str(python), str(Path(__file__).resolve()), "--runtime"]
    if args.reinstall:
        command.append("--reinstall")
    if args.doctor:
        command.append("--doctor")
    if args.no_browser:
        command.append("--no-browser")
    if args.certificate:
        command.append("--certificate")
    command.extend(["--port", str(args.port), "--ca-port", str(args.ca_port)])
    try:
        return subprocess.call(command, cwd=str(APP_DIR))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
