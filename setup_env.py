#!/usr/bin/env python3
"""
One-command setup for Spatial-Analysis-of-Cities (Python + R halves).

Usage (from the project root, with any Python 3.10+):

    python setup_env.py                  # full setup: venv + Python pkgs + R pkgs
    python setup_env.py --skip-r         # Python half only
    python setup_env.py --skip-python    # R half only
    python setup_env.py --gpu            # use the default (CUDA-capable) PyTorch build
    python setup_env.py --system-deps    # Debian/Ubuntu: apt-get the system libs R needs (uses sudo)
    python setup_env.py --run-tests      # run the offline tests afterwards
    python setup_env.py --dry-run        # show what would be done, change nothing

What it does
  1. Creates a virtual environment (.venv by default).
  2. Installs PyTorch (CPU build unless --gpu / macOS), then requirements.txt,
     then earthengine-api + requests (used by the GHSL R script via reticulate).
  3. Installs the R packages used by the R scripts:
       sf, terra, dplyr, ggplot2, osmextract, osmdata, here, reticulate
  4. Checks Java (only needed for the optional Spark engine) and verifies imports.

Safe to re-run: anything already installed is skipped.
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MIN_PY = (3, 10)
SYSTEM = platform.system()  # "Windows", "Darwin", "Linux"

R_PACKAGES = [
    "sf", "terra", "dplyr", "ggplot2",
    "osmextract", "osmdata", "here", "reticulate",
]

# System libraries that source builds of sf/terra/etc. need on Debian/Ubuntu.
APT_PACKAGES = [
    "r-base", "r-base-dev", "build-essential", "cmake", "pkg-config",
    "libgdal-dev", "libgeos-dev", "libproj-dev", "libudunits2-dev",
    "libsqlite3-dev", "libssl-dev", "libcurl4-openssl-dev", "libxml2-dev",
    "libfontconfig1-dev", "libharfbuzz-dev", "libfribidi-dev",
    "libfreetype6-dev", "libpng-dev", "libtiff5-dev", "libjpeg-dev",
    "default-jre-headless",
]

# Imports that must work after the Python install (module name -> pip name).
PY_CHECKS = {
    "numpy": "numpy", "pandas": "pandas", "scipy": "scipy",
    "sklearn": "scikit-learn", "matplotlib": "matplotlib",
    "geopandas": "geopandas", "shapely": "shapely", "pyogrio": "pyogrio",
    "pyproj": "pyproj", "rasterio": "rasterio", "statsmodels": "statsmodels",
    "torch": "torch", "umap": "umap-learn",
    "ee": "earthengine-api", "requests": "requests",
}
PY_OPTIONAL_CHECKS = {"pyspark": "pyspark", "pyarrow": "pyarrow"}

DRY = False


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def say(msg: str = "") -> None:
    print(msg, flush=True)


def step(title: str) -> None:
    say(f"\n=== {title} ===")


def run(cmd: list[str], check: bool = True, **kw) -> int:
    say("  $ " + " ".join(str(c) for c in cmd))
    if DRY:
        return 0
    proc = subprocess.run([str(c) for c in cmd], **kw)
    if check and proc.returncode != 0:
        raise SystemExit(f"Command failed (exit {proc.returncode}). See the output above.")
    return proc.returncode


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if SYSTEM == "Windows" else "bin/python")


# --------------------------------------------------------------------------- #
# Python half
# --------------------------------------------------------------------------- #
def ensure_gitignored(venv: Path) -> None:
    """Make sure the virtual environment folder is listed in .gitignore."""
    try:
        rel = venv.resolve().relative_to(ROOT).as_posix() + "/"
    except ValueError:
        return  # venv lives outside the project, nothing to ignore
    gi = ROOT / ".gitignore"
    lines = gi.read_text().splitlines() if gi.exists() else []
    if rel in lines or rel.rstrip("/") in lines:
        return
    say(f"  Adding {rel} to .gitignore so it is never committed")
    if not DRY:
        with gi.open("a") as fh:
            fh.write(("\n" if lines and lines[-1] else "") + rel + "\n")


def setup_python(venv: Path, gpu: bool) -> Path:
    step("Python environment")
    if sys.version_info < MIN_PY:
        raise SystemExit(
            f"Python {MIN_PY[0]}.{MIN_PY[1]}+ is required, found {platform.python_version()}."
        )
    say(f"  Using Python {platform.python_version()} ({sys.executable})")

    ensure_gitignored(venv)
    py = venv_python(venv)
    if py.exists():
        say(f"  Virtual environment already exists: {venv}")
    else:
        say(f"  Creating virtual environment: {venv}")
        rc = run([sys.executable, "-m", "venv", venv], check=False)
        if rc != 0:
            hint = ""
            if SYSTEM == "Linux":
                hint = "\n  On Debian/Ubuntu run:  sudo apt-get install python3-venv"
            raise SystemExit("Could not create the virtual environment." + hint)

    pip = [py, "-m", "pip"]
    run(pip + ["install", "--upgrade", "pip", "wheel", "setuptools"])

    step("PyTorch")
    already = (not DRY) and subprocess.run(
        [str(py), "-c", "import torch"], capture_output=True
    ).returncode == 0
    if already:
        say("  torch already installed, skipping.")
    elif gpu or SYSTEM == "Darwin":
        # macOS wheels have no CUDA, the default build is already small.
        run(pip + ["install", "torch>=2.0"])
    else:
        say("  Installing the CPU-only build (much smaller). Use --gpu for CUDA.")
        rc = run(
            pip + ["install", "torch>=2.0", "--index-url",
                   "https://download.pytorch.org/whl/cpu"],
            check=False,
        )
        if rc != 0:
            say("  CPU index failed, falling back to the default PyPI build.")
            run(pip + ["install", "torch>=2.0"])

    step("Python packages (requirements.txt)")
    run(pip + ["install", "-r", ROOT / "requirements.txt"])

    step("Earth Engine client (used by the GHSL R script)")
    run(pip + ["install", "earthengine-api", "requests"])
    return py


def verify_python(py: Path) -> bool:
    step("Verifying Python imports")
    if DRY:
        say("  (dry run, skipped)")
        return True
    code = textwrap.dedent(
        """
        import importlib, sys
        required = %r
        optional = %r
        bad = []
        for mod, pipname in required.items():
            try:
                importlib.import_module(mod)
            except Exception as e:
                bad.append((mod, pipname, e))
        for mod, pipname in optional.items():
            try:
                importlib.import_module(mod)
            except Exception as e:
                print(f"  [optional] {pipname}: not importable ({e})")
        for mod, pipname, e in bad:
            print(f"  [FAIL] {pipname}: {e}")
        if not bad:
            print("  All required Python packages import correctly.")
        sys.exit(1 if bad else 0)
        """
    ) % (PY_CHECKS, PY_OPTIONAL_CHECKS)
    return subprocess.run([str(py), "-c", code]).returncode == 0


# --------------------------------------------------------------------------- #
# R half
# --------------------------------------------------------------------------- #
R_INSTALL_SCRIPT = r"""
options(repos = c(CRAN = "https://cloud.r-project.org"), warn = 1)
pkgs <- c(%s)

# Make sure there is a writable library (user library if the system one is read-only).
user_lib <- Sys.getenv("R_LIBS_USER")
if (nzchar(user_lib)) {
  if (!dir.exists(user_lib)) dir.create(user_lib, recursive = TRUE, showWarnings = FALSE)
  .libPaths(c(user_lib, .libPaths()))
}
cat("R", as.character(getRversion()), "- installing into:", .libPaths()[1], "\n")

have <- function(p) requireNamespace(p, quietly = TRUE)
missing <- pkgs[!vapply(pkgs, have, logical(1))]
if (length(missing)) {
  cat("Installing:", paste(missing, collapse = ", "), "\n")
  ncpu <- max(1L, parallel::detectCores() - 1L)
  install.packages(missing, lib = .libPaths()[1], Ncpus = ncpu, dependencies = TRUE)
} else {
  cat("All R packages already installed.\n")
}

still <- pkgs[!vapply(pkgs, have, logical(1))]
if (length(still)) {
  cat("\nFAILED to install:", paste(still, collapse = ", "), "\n")
  quit(status = 1)
}
for (p in pkgs) cat(sprintf("  ok  %%-12s %%s\n", p, as.character(packageVersion(p))))
cat("R packages ready.\n")
"""


def find_rscript() -> str | None:
    found = shutil.which("Rscript")
    if found:
        return found
    if SYSTEM == "Windows":
        # Default CRAN install location, newest version first.
        for base in (os.environ.get("ProgramFiles", r"C:\Program Files"),
                     os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")):
            rdir = Path(base) / "R"
            if rdir.is_dir():
                for cand in sorted(rdir.glob("R-*"), reverse=True):
                    exe = cand / "bin" / "Rscript.exe"
                    if exe.exists():
                        return str(exe)
    return None


def r_install_help() -> str:
    if SYSTEM == "Windows":
        return ("Install R from https://cran.r-project.org/bin/windows/base/ "
                "(or: winget install RProject.R), then re-run this script.")
    if SYSTEM == "Darwin":
        return "Install R with:  brew install r   (or from https://cran.r-project.org/bin/macosx/)"
    return ("Install R with:  sudo apt-get install r-base r-base-dev   "
            "(or re-run with --system-deps).")


def apt_system_deps() -> None:
    step("System libraries (apt)")
    if SYSTEM != "Linux" or not shutil.which("apt-get"):
        say("  --system-deps only supports Debian/Ubuntu (apt-get). Skipping.")
        return
    sudo = [] if (hasattr(os, "geteuid") and os.geteuid() == 0) else ["sudo"]
    run(sudo + ["apt-get", "update"], check=False)
    run(sudo + ["apt-get", "install", "-y"] + APT_PACKAGES)


def setup_r() -> bool:
    step("R packages")
    rscript = find_rscript()
    if not rscript:
        say("  Rscript was not found on PATH.")
        say("  " + r_install_help())
        return False
    say(f"  Using {rscript}")
    code = R_INSTALL_SCRIPT % ", ".join(f'"{p}"' for p in R_PACKAGES)
    if DRY:
        say("  (dry run) would install: " + ", ".join(R_PACKAGES))
        return True
    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False) as fh:
        fh.write(code)
        tmp = fh.name
    try:
        rc = subprocess.run([rscript, "--vanilla", tmp]).returncode
    finally:
        os.unlink(tmp)
    if rc != 0:
        say("\n  Some R packages failed to build.")
        if SYSTEM == "Linux":
            say("  Usually a missing system library. Re-run with --system-deps "
                "(Debian/Ubuntu) or install GDAL, GEOS, PROJ, UDUNITS and libcurl/openssl dev packages.")
        elif SYSTEM == "Windows":
            say("  On Windows CRAN ships binaries, so check your internet/proxy; "
                "if R is older than 4.2 please update R.")
        return False
    return True


# --------------------------------------------------------------------------- #
# Java (optional)
# --------------------------------------------------------------------------- #
def check_java() -> None:
    step("Java (optional, only for --engine spark)")
    java = shutil.which("java")
    if not java:
        say("  Java not found. Only needed for the Spark engine.")
        say("  Install Java 17+ (e.g. Temurin from https://adoptium.net) if you want it.")
        return
    out = subprocess.run([java, "-version"], capture_output=True, text=True)
    first = (out.stderr or out.stdout).splitlines()[0] if (out.stderr or out.stdout) else "?"
    say(f"  Found: {first}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    global DRY
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--venv", default=str(ROOT / ".venv"), help="virtual environment path (default: .venv)")
    ap.add_argument("--skip-python", action="store_true", help="do not set up the Python half")
    ap.add_argument("--skip-r", action="store_true", help="do not set up the R half")
    ap.add_argument("--gpu", action="store_true", help="install the default (CUDA-capable) PyTorch build")
    ap.add_argument("--system-deps", action="store_true", help="apt-get system libraries first (Debian/Ubuntu, sudo)")
    ap.add_argument("--run-tests", action="store_true", help="run the offline Python tests at the end")
    ap.add_argument("--dry-run", action="store_true", help="print actions without changing anything")
    args = ap.parse_args()
    DRY = args.dry_run

    say(f"Project: {ROOT}")
    say(f"Platform: {SYSTEM} {platform.machine()}")

    ok = True
    py = None
    if args.system_deps:
        apt_system_deps()

    if not args.skip_python:
        py = setup_python(Path(args.venv).resolve(), args.gpu)
        ok &= verify_python(py)
        check_java()

    if not args.skip_r:
        ok &= setup_r()

    if args.run_tests and py is not None and not DRY:
        step("Offline tests")
        ok &= subprocess.run([str(py), "run_analysis.py", "--test"], cwd=ROOT).returncode == 0

    step("Done" if ok else "Finished with problems (see above)")
    if py is not None:
        act = (f"{Path(args.venv).name}\\Scripts\\activate" if SYSTEM == "Windows"
               else f"source {Path(args.venv).name}/bin/activate")
        say("  Activate the Python environment:   " + act)
        say("  Check the Python half:             python run_analysis.py --test")
    if not args.skip_r:
        say("  Check the R half:                  Rscript run_pipeline.R --check")
    say("  (GHSL also needs a one-time Earth Engine login, see data_extraction/ghsl_builtup_capitals.R)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
