"""Capture real Chrome document regions through its local debugging socket."""
from __future__ import annotations

import base64
from hashlib import sha256
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import tempfile
import time
from urllib.parse import urlsplit
from urllib.request import urlopen


class DebugSocket:
    """Small localhost-only WebSocket client for sequential CDP requests."""

    def __init__(self, url):
        address = urlsplit(url)
        if address.scheme != "ws" or address.hostname not in ("localhost", "127.0.0.1"):
            raise ValueError("Only the isolated local Chrome debugging socket is supported")
        self.socket = socket.create_connection((address.hostname, address.port), timeout=30)
        self.buffer = bytearray()
        self.sequence = 0
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        request = (
            f"GET {address.path} HTTP/1.1\r\nHost: {address.netloc}\r\n"
            "Upgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Version: 13\r\n"
            f"Sec-WebSocket-Key: {key}\r\n\r\n"
        )
        self.socket.sendall(request.encode("ascii"))
        response = bytearray()
        while b"\r\n\r\n" not in response:
            response.extend(self.socket.recv(4096))
        header, rest = response.split(b"\r\n\r\n", 1)
        if not header.startswith(b"HTTP/1.1 101"):
            raise RuntimeError("Chrome rejected its local WebSocket connection")
        self.buffer.extend(rest)

    def read(self, count):
        while len(self.buffer) < count:
            chunk = self.socket.recv(max(4096, count - len(self.buffer)))
            if not chunk:
                raise EOFError("Chrome debugging socket closed")
            self.buffer.extend(chunk)
        result = bytes(self.buffer[:count])
        del self.buffer[:count]
        return result

    def send(self, payload, opcode=1):
        payload = payload if isinstance(payload, bytes) else payload.encode("utf-8")
        mask = os.urandom(4)
        length = len(payload)
        header = bytes([0x80 | opcode])
        if length < 126:
            header += bytes([0x80 | length])
        elif length < 65536:
            header += bytes([0x80 | 126]) + struct.pack("!H", length)
        else:
            header += bytes([0x80 | 127]) + struct.pack("!Q", length)
        self.socket.sendall(header + mask + bytes(value ^ mask[index % 4] for index, value in enumerate(payload)))

    def receive(self):
        fragments = []
        while True:
            first, second = self.read(2)
            size = second & 127
            if size == 126:
                size = struct.unpack("!H", self.read(2))[0]
            elif size == 127:
                size = struct.unpack("!Q", self.read(8))[0]
            mask = self.read(4) if second & 128 else None
            payload = self.read(size)
            if mask:
                payload = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
            opcode = first & 15
            if opcode == 8:
                raise EOFError("Chrome debugging socket closed")
            if opcode == 9:
                self.send(payload, opcode=10)
                continue
            if opcode in (0, 1):
                fragments.append(payload)
                if first & 128:
                    return json.loads(b"".join(fragments))

    def call(self, method, params=None):
        self.sequence += 1
        self.send(json.dumps({"id": self.sequence, "method": method, "params": params or {}}))
        while True:
            response = self.receive()
            if response.get("id") != self.sequence:
                continue
            if "error" in response:
                raise RuntimeError(str(response["error"]))
            return response.get("result", {})

    def evaluate(self, expression):
        result = self.call("Runtime.evaluate", {"expression": expression, "awaitPromise": True, "returnByValue": True})
        if "exceptionDetails" in result:
            raise RuntimeError(str(result["exceptionDetails"]))
        return result.get("result", {}).get("value")


def render_screenshots(report, folder, browser):
    report, folder = Path(report).resolve(), Path(folder).resolve()
    original = report.read_bytes()
    profile = Path(tempfile.mkdtemp(prefix="chrome-visual-cdp-", dir=folder))
    command = [
        str(browser), "--headless=new", "--disable-gpu", "--no-first-run",
        "--no-default-browser-check", "--disable-default-apps", "--disable-extensions",
        "--disable-background-networking", "--disable-component-update", "--disable-sync",
        "--allow-file-access-from-files", "--window-size=1440,1200",
        "--remote-debugging-address=127.0.0.1", "--remote-debugging-port=0",
        "--user-data-dir=" + str(profile), "about:blank",
    ]
    results, debugger = [], None
    with (folder / "browser_visual_chrome.stderr.log").open("wb") as stderr:
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=stderr, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            deadline = time.monotonic() + 30
            port_file = profile / "DevToolsActivePort"
            while not port_file.is_file():
                if time.monotonic() > deadline or process.poll() is not None:
                    raise RuntimeError("Isolated screenshot Chrome did not start")
                time.sleep(.05)
            port = port_file.read_text(encoding="utf-8").splitlines()[0]
            with urlopen("http://127.0.0.1:" + port + "/json/list", timeout=10) as response:
                target = next(item for item in json.load(response) if item.get("type") == "page")
            debugger = DebugSocket(target["webSocketDebuggerUrl"])
            debugger.call("Page.enable")
            debugger.call("Emulation.setDeviceMetricsOverride", {"width": 1440, "height": 1200, "deviceScaleFactor": 1, "mobile": False})
            debugger.call("Page.navigate", {"url": report.as_uri() + "#overview"})
            deadline = time.monotonic() + 60
            while not debugger.evaluate("Boolean(window.MultigridReport && document.documentElement.dataset.reportReady === 'true')"):
                if time.monotonic() > deadline:
                    raise RuntimeError("Report did not initialize for screenshot capture")
                time.sleep(.1)
            for name, page, anchor in (
                ("overview", "overview", None), ("explorer", "explorer", None),
                ("explorer_convergence", "explorer", "residual-chart"),
                ("explorer_ledgers", "explorer", "sequence-stage-table"),
            ):
                settings = json.dumps({"page": page, "anchor": anchor})
                region = debugger.evaluate("""(async () => {
                  const settings = SETTINGS;
                  MultigridReport.setTheme('light'); MultigridReport.navigate(settings.page);
                  scrollTo(0,0);
                  await new Promise(resolve => setTimeout(resolve,300));
                  await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
                  const target=settings.anchor && document.getElementById(settings.anchor);
                  const box=target && (target.closest('.plot-grid,.table-wrap') || target).getBoundingClientRect();
                  return {x:0,y:box?box.top+scrollY:0,width:1440,height:1200,scale:1};
                })()""".replace("SETTINGS", settings))
                capture = debugger.call("Page.captureScreenshot", {"format": "png", "fromSurface": True, "captureBeyondViewport": True, "clip": region})
                image = folder / (name + ".png")
                image.write_bytes(base64.b64decode(capture["data"]))
                results.append({"image": image.name, "page": page, "anchor": anchor, "clip": region, "sha256": sha256(image.read_bytes()).hexdigest(), "visual_inspection_performed": False})
        finally:
            if debugger is not None:
                try:
                    debugger.call("Browser.close")
                except (OSError, EOFError):
                    pass
                debugger.socket.close()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=10)
    metadata = {"screenshots": results, "actual_browser": True, "capture": "CDP document regions from the original offline report", "report_sha256": sha256(original).hexdigest(), "report_unchanged": report.read_bytes() == original, "isolated_profile": str(profile)}
    (folder / "browser_screenshots.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return results
