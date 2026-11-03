"""Check local runtime requirements and download the requested models."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

OLLAMA_MODELS = ("qwen2.5:7b-instruct", "gemma3:12b")
EMBEDDING_MODEL = "BAAI/bge-m3"


def run_ollama(args: list[str], *, capture: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ollama", *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )


def ollama_list() -> tuple[set[str], dict[str, str]]:
    result = run_ollama(["list"])
    if result.returncode:
        raise RuntimeError(result.stdout or "ollama list failed")
    names: set[str] = set()
    sizes: dict[str, str] = {}
    for line in (result.stdout or "").splitlines()[1:]:
        fields = line.split()
        if len(fields) >= 3:
            names.add(fields[0])
            sizes[fields[0]] = " ".join(fields[2:])
    return names, sizes


def ensure_ollama_server() -> tuple[set[str], dict[str, str]]:
    try:
        return ollama_list()
    except (OSError, RuntimeError):
        print("Ollama server is not responding; starting `ollama serve`.")
        creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if platform.system() == "Windows" else 0
        subprocess.Popen(
            ["ollama", "serve"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creation_flags,
            start_new_session=platform.system() != "Windows",
        )
        for _ in range(30):
            time.sleep(2)
            try:
                return ollama_list()
            except (OSError, RuntimeError):
                continue
        raise RuntimeError("Could not start the Ollama server; `ollama list` still fails.")


def ensure_embedding_model() -> tuple[str, str]:
    try:
        from huggingface_hub import snapshot_download, try_to_load_from_cache
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required; install requirements.txt first.") from exc

    try:
        cached_config = try_to_load_from_cache(EMBEDDING_MODEL, "config.json")
        if not cached_config:
            raise FileNotFoundError("BGE-M3 config is not present in the Hugging Face cache.")
        path = snapshot_download(repo_id=EMBEDDING_MODEL, local_files_only=True)
        total_bytes = sum(item.stat().st_size for item in Path(path).rglob("*") if item.is_file())
        size = f"{total_bytes / (1024 ** 3):.2f} GiB"
        return "found", size
    except Exception as cache_error:
        print(f"BGE-M3 is not fully cached ({cache_error}); downloading the model (about 2.3 GB).")
        path = snapshot_download(repo_id=EMBEDDING_MODEL)
        total_bytes = sum(item.stat().st_size for item in Path(path).rglob("*") if item.is_file())
        size = f"{total_bytes / (1024 ** 3):.2f} GiB"
        return "downloaded", size


def main() -> int:
    if shutil.which("ollama") is None:
        print("Ollama is not installed. Install it from https://ollama.com, then run this script again.")
        return 1

    version = run_ollama(["--version"])
    if version.returncode:
        print(version.stdout or "Unable to run `ollama --version`.")
        return 1
    print(version.stdout.strip())

    try:
        installed, sizes = ensure_ollama_server()
    except (OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}")
        return 1

    statuses: list[tuple[str, str, str]] = []
    for model in OLLAMA_MODELS:
        if model in installed:
            statuses.append((model, "found", sizes.get(model, "unknown")))
            continue
        print(f"Downloading {model}. Model download sizes vary by Ollama build and quantization.")
        result = run_ollama(["pull", model], capture=False)
        if result.returncode:
            print(f"ERROR: `ollama pull {model}` failed with exit code {result.returncode}.")
            return result.returncode
        try:
            installed, sizes = ollama_list()
        except (OSError, RuntimeError) as exc:
            print(f"ERROR: Unable to verify {model}: {exc}")
            return 1
        if model not in installed:
            print(f"ERROR: {model} is still absent after the pull.")
            return 1
        statuses.append((model, "downloaded", sizes.get(model, "unknown")))

    try:
        embedding_status, embedding_path = ensure_embedding_model()
    except Exception as exc:
        print(f"ERROR: Could not prepare {EMBEDDING_MODEL}: {exc}")
        return 1
    statuses.append((EMBEDDING_MODEL, embedding_status, embedding_path))

    print("\nModel status")
    print(f"{'Model':<28} {'Status':<12} Size / location")
    print("-" * 84)
    for model, status, size in statuses:
        print(f"{model:<28} {status:<12} {size}")

    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        print("GPU status: nvidia-smi was not found.")
        return 0
    gpu = subprocess.run(
        [nvidia_smi, "--query-gpu=name,memory.free", "--format=csv,noheader"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if gpu.returncode:
        print(f"GPU status unavailable: {gpu.stdout.strip()}")
        return 0
    print(f"\nGPU free VRAM:\n{gpu.stdout.strip()}")
    free_values = []
    for line in gpu.stdout.splitlines():
        try:
            free_values.append(float(line.rsplit(",", 1)[1].strip().split()[0]))
        except (IndexError, ValueError):
            continue
    if free_values and min(free_values) < 8192:
        print("WARNING: Less than 8 GiB of VRAM is free.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
