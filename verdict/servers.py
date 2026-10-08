"""Server lifecycle: the "do it for them" layer.

verdict is the client side -- each model runs as its own server process.
This module collapses the whole setup dance into one command::

    verdict up von        # install (if needed) -> launch -> wait for port
                          # -> pin URL into verdict.yaml

URL resolution order (automatic, everywhere): explicit kwarg/flag >
``<NAME>_BASE_URL`` env var > server started by ``verdict up`` >
registry default. So after ``verdict up von``, plain
``verdict decide --backend von ...`` just works -- no flags, no YAML edits.

State lives in ``~/.verdict/servers/<backend>/server.json`` (pid, url,
port) plus ``server.log``.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse


def home() -> Path:
    d = Path.home() / ".verdict" / "servers"
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class ServerRecipe:
    backend: str
    install: list[str] = field(default_factory=list)
    serve: str = ""
    port: int = 8000
    host: str = "127.0.0.1"
    env_var: str = ""
    auto: bool = True
    manual: str = ""
    notes: str = ""

    @property
    def default_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1/systemone"


# {dir} = ~/.verdict/servers/<backend>, {port} = recipe port,
# {python} = current interpreter (so pip installs land where verdict runs).
RECIPES: dict[str, ServerRecipe] = {
    "kev": ServerRecipe(
        backend="kev",
        install=[
            'git clone https://github.com/jaredpalmer/kev "{dir}/src"',
            '"{python}" -m pip install -q -e "{dir}/src[serve]"',
        ],
        serve='cd "{dir}/src" && "{python}" -m kev.serve '
              "--run jaredpalmer/kev-0.8b --port {port}",
        port=8009,
        env_var="KEV_BASE_URL",
        notes="First run downloads ~2GB of weights from HuggingFace.",
    ),
    "strands": ServerRecipe(
        backend="strands",
        install=['"{python}" -m pip install -q strands-decider'],
        serve="strands-decider serve",
        port=8013,
        env_var="STRANDS_BASE_URL",
        notes="Binds 127.0.0.1 with no auth -- add your own for shared machines. "
              "Port 8000 assumed; override with STRANDS_BASE_URL if yours differs.",
    ),
    "laya": ServerRecipe(
        backend="laya",
        install=['"{python}" -m pip install -q "laya[serve]"'],
        serve="laya-serve",
        port=8000,
        env_var="LAYA_BASE_URL",
        notes="Serves the base English checkpoint; pick checkpoints via laya's own config.",
    ),
    "opendecider": ServerRecipe(
        backend="opendecider",
        install=['"{python}" -m pip install -q opendecider'],
        serve="opendecider serve",
        port=8014,
        env_var="OPENDECIDER_BASE_URL",
        notes="Port 8000 assumed; override with OPENDECIDER_BASE_URL if yours differs.",
    ),
    "von": ServerRecipe(
        backend="von",
        install=['"{python}" -m pip install -q von-sdk'],
        serve="von serve --port {port}",
        port=8017,
        env_var="VON_BASE_URL",
        notes="395M -- the fastest first real model. CPU-friendly.",
    ),
    "mapika": ServerRecipe(
        backend="mapika",
        install=[
            'git clone https://github.com/Mapika/decider "{dir}/src"',
            '"{python}" -m pip install -q -e "{dir}/src"',
        ],
        serve='bash "{dir}/src/scripts/serve.sh"',
        port=8018,
        env_var="MAPIKA_BASE_URL",
        notes="serve.sh port assumed 8000; override with MAPIKA_BASE_URL if yours differs.",
    ),
    # -- manual: no reliable automatic recipe; `verdict up` prints guidance --
    "clef": ServerRecipe(
        backend="clef", auto=False, env_var="CLEF_BASE_URL",
        manual="Fastest path is hosted: Cloudflare Workers AI @cf/cloudflare/clef "
               "($0.24/M input). Point CLEF_BASE_URL at your worker URL.\n"
               "Self-hosted: open-jevlike-infer (vLLM + custom scoring head):\n"
               "  git clone https://github.com/Arcobalneo/open-jevlike-infer",
    ),
    "clef-flash": ServerRecipe(
        backend="clef-flash", auto=False, env_var="CLEF_FLASH_BASE_URL",
        manual="Fastest path is hosted: Cloudflare Workers AI @cf/cloudflare/clef-flash "
               "($0.09/M input). Point CLEF_FLASH_BASE_URL at your worker URL.\n"
               "Self-hosted options: open-jevlike-infer, or the 4-bit MLX quant "
               "mlx-community/clef-flash-4bit (6.2GB, fits a 16GB Mac).",
    ),
    "wald": ServerRecipe(
        backend="wald", auto=False, env_var="WALD_BASE_URL",
        manual="Clone https://github.com/org2ai/Wald-4B (weights: org2ai/Wald-4B) "
               "and run its server; then WALD_BASE_URL=http://host:port/v1/systemone",
    ),
    "openjev": ServerRecipe(
        backend="openjev", auto=False, env_var="OPENJEV_BASE_URL",
        manual="Clone https://github.com/ejhshen/OpenJev and run its local "
               "/v1/systemone server; then OPENJEV_BASE_URL=http://host:port/v1/systemone",
    ),
    "clm": ServerRecipe(
        backend="clm", auto=False, env_var="CLM_BASE_URL",
        manual="See the Contrastive-LM repo for serving; then "
               "CLM_BASE_URL=http://host:port/v1/systemone",
    ),
    "semif": ServerRecipe(
        backend="semif", auto=False, env_var="SEMIF_MODEL",
        manual="SemIf is a local harness, not a server: install the semif-score CLI "
               "and weights per https://github.com/TheoLeeCJ/SemIf, then use it directly.",
    ),
    "jev": ServerRecipe(
        backend="jev", auto=False, env_var="TYPESAFE_API_KEY",
        manual='pip install -e ".[jev]" and set TYPESAFE_API_KEY.',
    ),
    "mock": ServerRecipe(
        backend="mock", auto=False,
        manual="mock is in-process -- nothing to start.",
    ),
}


def probe(url: str, timeout: float = 2.0) -> bool:
    """Is something listening at this URL? TCP-level: port-open, not model-ready."""
    try:
        p = urlparse(url)
        host, port = p.hostname or "127.0.0.1", p.port or 80
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def managed_url(backend: str) -> str | None:
    """URL of a `verdict up`-managed server, if it's still alive."""
    f = home() / backend / "server.json"
    if not f.exists():
        return None
    try:
        info = json.loads(f.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    try:
        os.kill(info["pid"], 0)
    except (OSError, KeyError):
        return None
    return info.get("url")


def resolve_url(backend: str, explicit: str | None = None) -> str | None:
    """Automatic URL resolution: explicit > env > managed server > recipe default."""
    if explicit:
        return explicit
    recipe = RECIPES.get(backend)
    if recipe and recipe.env_var:
        val = os.environ.get(recipe.env_var, "")
        if val.startswith(("http://", "https://")):
            return val
    url = managed_url(backend)
    if url:
        return url
    if recipe and recipe.auto:
        return recipe.default_url
    return None


class ServerManager:
    """Start/stop/inspect one backend's server process."""

    def __init__(self, backend: str):
        if backend not in RECIPES:
            raise KeyError(f"unknown backend {backend!r}")
        self.backend = backend
        self.recipe = RECIPES[backend]
        self.dir = home() / backend
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.dir / "server.json"
        self.log_file = self.dir / "server.log"

    # -- state ---------------------------------------------------------

    def _read_state(self) -> dict | None:
        if not self.state_file.exists():
            return None
        try:
            return json.loads(self.state_file.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def is_running(self) -> bool:
        st = self._read_state()
        if not st:
            return False
        try:
            os.kill(st["pid"], 0)
            return True
        except OSError:
            return False

    # -- lifecycle -----------------------------------------------------

    def start(self, no_install: bool = False, wait_s: int = 180) -> str:
        """Install (unless no_install), launch, wait for the port. Returns the URL."""
        r = self.recipe
        if not r.auto:
            raise RuntimeError(f"no automatic setup for {self.backend!r}:\n{r.manual}")
        url = r.default_url

        if self.is_running():
            if probe(url):
                return url
            self.stop()  # stale pid file; start fresh

        if not no_install:
            self._install()
        self._launch()
        if not self._wait_for_port(url, wait_s):
            raise RuntimeError(
                f"{self.backend} didn't open {url} within {wait_s}s. "
                f"Check {self.log_file} -- first run downloads weights, which can "
                f"take a few minutes; the server may still be starting."
            )
        # Port-open is not model-ready: some servers (von) finish loading
        # weights after accepting connections. One tiny real request proves
        # the model answers before we declare victory.
        if not self._wait_for_ready(url, wait_s=120):
            print(f"warning: {url} is listening but the model isn't answering yet "
                  f"(still loading weights?). Check {self.log_file}.")
        return url

    def _env(self) -> dict:
        # Console scripts installed by the recipes (e.g. `von serve`) land
        # in the venv's bin -- make sure the subprocess can find them.
        env = dict(os.environ)
        bindir = str(Path(sys.executable).parent)
        env["PATH"] = bindir + os.pathsep + env.get("PATH", "")
        # Model installs (torch wheels etc.) can exceed small /tmp mounts;
        # give pip a roomy temp dir under our state dir.
        env.setdefault("TMPDIR", str(self.dir / "tmp"))
        (self.dir / "tmp").mkdir(parents=True, exist_ok=True)
        return env

    def _install(self) -> None:
        r = self.recipe
        ctx = {"dir": str(self.dir), "port": r.port, "python": sys.executable}
        for cmd in r.install:
            rendered = cmd.format(**ctx)
            print(f"  $ {rendered}")
            rc = subprocess.run(rendered, shell=True, env=self._env()).returncode
            if rc != 0:
                raise RuntimeError(
                    f"install failed for {self.backend} (exit {rc}). "
                    f"Run the commands manually, then `verdict up --no-install "
                    f"{self.backend}`."
                )

    def _launch(self) -> None:
        r = self.recipe
        ctx = {"dir": str(self.dir), "port": r.port, "python": sys.executable}
        log = open(self.log_file, "a")
        proc = subprocess.Popen(
            r.serve.format(**ctx), shell=True,
            stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, env=self._env(),
            # Stable working directory: servers resolve relative paths
            # (checkpoint dirs, caches) against CWD. Inheriting the caller's
            # CWD makes behavior depend on where `verdict up` was run.
            cwd=str(self.dir),
        )
        self.state_file.write_text(json.dumps({
            "pid": proc.pid, "url": r.default_url, "port": r.port,
        }))

    def _wait_for_port(self, url: str, wait_s: int) -> bool:
        t0 = time.time()
        while time.time() - t0 < wait_s:
            if probe(url, timeout=1.0):
                return True
            time.sleep(1.0)
        return False

    def _wait_for_ready(self, url: str, wait_s: int) -> bool:
        """POST a minimal choice question until the model answers HTTP 200."""
        import json
        import urllib.request

        payload = json.dumps({
            "state": "readiness probe",
            "model": self.backend,
            "questions": {
                "probe": {
                    "type": "choice",
                    "instructions": "Pick one.",
                    "criteria": {"a": "Option A", "b": "Option B"},
                }
            },
        }).encode()
        t0 = time.time()
        while time.time() - t0 < wait_s:
            try:
                req = urllib.request.Request(
                    url, data=payload, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if resp.status == 200:
                        return True
            except Exception:
                pass
            time.sleep(2.0)
        return False

    def stop(self) -> bool:
        """Stop the managed server. Returns True if something was stopped."""
        st = self._read_state()
        if not st:
            return False
        pid = st["pid"]
        try:
            # The server is launched via a shell (shell=True) as a process
            # group leader (start_new_session=True): kill the whole group so
            # the actual server child dies too, not just the shell.
            os.killpg(pid, signal.SIGTERM)
            for _ in range(10):
                time.sleep(0.5)
                try:
                    os.kill(pid, 0)
                except OSError:
                    break
            else:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except OSError:
                    pass
        except OSError:
            pass
        try:
            self.state_file.unlink()
        except OSError:
            pass
        return True


# -- verdict.yaml surgery (line-based: preserves comments and formatting) --

def wire_yaml(path: Path, backend: str, url: str) -> None:
    """Pin backends.<backend>.base_url in a verdict.yaml, preserving the rest."""
    path = Path(path)
    if not path.exists():
        path.write_text(
            "version: 1\n\n"
            f"backends:\n  {backend}:\n    base_url: {url}\n\n"
            "decisions: []\n"
        )
        return
    lines = path.read_text().splitlines(keepends=True)
    out: list[str] = []
    i, n = 0, len(lines)
    in_backends = False
    backend_seen = False
    url_written = False
    while i < n:
        line = lines[i]
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if indent == 0 and stripped.startswith("backends:"):
            in_backends = True
            out.append(line)
            i += 1
            continue
        if indent == 0 and stripped and in_backends:
            # leaving the backends block: insert missing entry at its end
            if not backend_seen:
                out.append(f"  {backend}:\n")
                out.append(f"    base_url: {url}\n")
                url_written = True
            in_backends = False
            out.append(line)
            i += 1
            continue
        if in_backends and indent == 2 and stripped.rstrip(":") == backend \
                and stripped.endswith(":"):
            backend_seen = True
            out.append(line)
            i += 1
            # scan this backend's sub-block for base_url
            while i < n:
                sub = lines[i]
                sub_indent = len(sub) - len(sub.lstrip())
                if sub.strip() and sub_indent <= 2:
                    break
                if sub_indent == 4 and sub.strip().startswith("base_url:"):
                    out.append(f"    base_url: {url}\n")
                    url_written = True
                else:
                    out.append(sub)
                i += 1
            if not url_written:
                out.append(f"    base_url: {url}\n")
                url_written = True
            continue
        out.append(line)
        i += 1
    if in_backends and not backend_seen:
        out.append(f"  {backend}:\n")
        out.append(f"    base_url: {url}\n")
    # file had no backends section at all
    if not any(l.strip().startswith("backends:") for l in lines):
        if out and not out[-1].endswith("\n"):
            out[-1] += "\n"
        out.append(f"\nbackends:\n  {backend}:\n    base_url: {url}\n")
    path.write_text("".join(out))


def set_policy_backend(path: Path, backend: str) -> int:
    """Point every decision's policy.backend at <backend>. Returns count changed."""
    lines = Path(path).read_text().splitlines(keepends=True)
    changed = 0
    for i, line in enumerate(lines):
        if line.startswith("      backend:"):
            lines[i] = f"      backend: {backend}\n"
            changed += 1
    Path(path).write_text("".join(lines))
    return changed
