# PlatformIO pre-script: wifi.yaml -> include/wifi_secrets.gen.h
# Parse top-level ssid / password (or pass) without PyYAML.

from pathlib import Path

Import("env")  # noqa: F821

ROOT = Path(env["PROJECT_DIR"])  # noqa: F821
YAML_PATH = ROOT / "wifi.yaml"
HEADER_PATH = ROOT / "include" / "wifi_secrets.gen.h"


def _c_escape(text):
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _unquote(value):
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def load_wifi_yaml(path):
    ssid = ""
    password = ""
    if not path.is_file():
        return ssid, password
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower()
        value = _unquote(value)
        if key == "ssid":
            ssid = value
        elif key in ("password", "pass"):
            password = value
    return ssid, password


ssid, password = load_wifi_yaml(YAML_PATH)
HEADER_PATH.parent.mkdir(parents=True, exist_ok=True)
HEADER_PATH.write_text(
    "#pragma once\n"
    "// generated from wifi.yaml; do not edit\n"
    f'#define WIFI_SSID "{_c_escape(ssid)}"\n'
    f'#define WIFI_PASS "{_c_escape(password)}"\n',
    encoding="utf-8",
)
print(f"wifi.yaml -> {HEADER_PATH.name}  ssid_set={bool(ssid)}")
