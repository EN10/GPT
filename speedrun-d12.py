import os
import subprocess
from pathlib import Path
import modal

APP_NAME = "nanochat-speedrun-h100"
VOLUME_NAME = "nanochat-persistent-storage"
GPU_CONFIG = "H100:8"

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

VOL_PATH = Path("/vol")
RUNS_DIR = VOL_PATH / "runs"
DATA_DIR = VOL_PATH / "data"

# CUDA base image (keep your current working one); key fix is installing cuSPARSELt into the uv venv.
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "curl", "wget", "build-essential", "pkg-config", "findutils")
    .pip_install("uv")
)

def _run(cmd: str, cwd: Path | None = None, env: dict | None = None) -> None:
    base_env = os.environ.copy()
    if env:
        base_env.update(env)

    # Helpful on CUDA images
    base_env["LD_LIBRARY_PATH"] = base_env.get("LD_LIBRARY_PATH", "") + ":/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu"

    print(f"\n[EXEC] {cmd}\n", flush=True)
    subprocess.run(["bash", "-lc", cmd], cwd=cwd, env=base_env, check=True)

def _find_file(name: str, root: Path) -> Path | None:
    try:
        out = subprocess.check_output(["find", str(root), "-name", name], text=True).strip().splitlines()
        if out and out[0]:
            return Path(sorted(out, key=len)[0])
    except Exception:
        return None
    return None

def _setup_repo(repo_ref: str, repo_url: str) -> Path:
    workdir = Path("/root/nanochat_work")
    repo_dir = workdir / "nanochat"

    _run(f"mkdir -p '{RUNS_DIR}' '{DATA_DIR}'")
    if not repo_dir.exists():
        _run(f"mkdir -p '{workdir}'")
        _run(f"git clone --depth 1 '{repo_url}' '{repo_dir}'")

    _run("git fetch --all --tags --prune", cwd=repo_dir)
    _run(f"git checkout '{repo_ref}'", cwd=repo_dir)

    # Persist datasets + logs/checkpoints to the volume
    _run(f"rm -rf data && ln -s '{DATA_DIR}' data", cwd=repo_dir)
    _run(f"rm -rf logs && ln -s '{RUNS_DIR}' logs", cwd=repo_dir)

    # Make speedrun available at repo root for convenience
    sr = _find_file("speedrun.sh", repo_dir)
    if sr and sr.exists():
        _run(f"chmod +x '{sr}'", cwd=repo_dir)
        if sr.parent.name == "runs" and not (repo_dir / "speedrun.sh").exists():
            _run("cp runs/speedrun.sh ./speedrun.sh && chmod +x ./speedrun.sh", cwd=repo_dir)

    return repo_dir

def _ensure_uv_env_has_cuda_bits(repo_dir: Path) -> None:
    """
    Create/sync the repo .venv, then install cuSPARSELt which PyTorch may require at import time.
    """
    # 1) Create/sync the venv once
    _run("uv sync --inexact", cwd=repo_dir)

    # 2) Install cuSPARSELt into the same .venv used by uv (fixes libcusparseLt.so.0) [web:126]
    _run("uv pip install nvidia-cusparselt-cu12", cwd=repo_dir)

def _uv_run(repo_dir: Path, cmd: str) -> None:
    """
    Run inside the project env but avoid re-syncing each time. [web:137]
    Falls back if --no-sync isn’t supported by your uv version.
    """
    try:
        _run(f"uv run --no-sync {cmd}", cwd=repo_dir)  # [web:137]
    except subprocess.CalledProcessError:
        _run(f"uv run {cmd}", cwd=repo_dir)

def _patch_speedrun_for_checkpoints(repo_dir: Path, eval_interval: int = 100) -> None:
    """
    Best-effort: append checkpointing flags to training invocations inside speedrun.sh.
    (If patterns don’t match, it does nothing.)
    """
    sr = repo_dir / "speedrun.sh"
    if not sr.exists():
        return

    flags = f" --eval_interval={eval_interval} --always_save_checkpoint=True"
    # Patch lines that invoke training scripts/modules.
    for needle in ["scripts.base_train", "scripts.mid_train", "scripts.chat_sft", "python -m", "uv run"]:
        _run(f"grep -q '{needle}' speedrun.sh && sed -i '/{needle}/ s/$/{flags}/' speedrun.sh || true", cwd=repo_dir)

@app.function(
    image=image,
    gpu=GPU_CONFIG,
    timeout=24 * 60 * 60,
    volumes={str(VOL_PATH): vol},
)
def run_speedrun(repo_ref: str = "master", model: str = "d12", force_restart: bool = False):
    repo_dir = _setup_repo(repo_ref=repo_ref, repo_url="https://github.com/karpathy/nanochat.git")

    # Ensure torch can import by installing cuSPARSELt into the uv venv. [web:126]
    _ensure_uv_env_has_cuda_bits(repo_dir)

    # Force checkpointing every 100 steps in speedrun if possible (patch script),
    # and also pass flags when we directly invoke base_train.
    _patch_speedrun_for_checkpoints(repo_dir, eval_interval=100)

    if force_restart:
        _run(f"rm -rf '{RUNS_DIR}/{model}'", cwd=repo_dir)

    # Run the speedrun (now located at repo root because we copied it from runs/ if needed) [web:43]
    _run(f"./speedrun.sh {model}", cwd=repo_dir)

    vol.commit()
    return f"Done. Outputs in {RUNS_DIR}"

@app.function(
    image=image,
    gpu=GPU_CONFIG,
    timeout=30 * 60,
    volumes={str(VOL_PATH): vol},
)
def smoke_test_10_steps(repo_ref: str = "master", model: str = "d12"):
    repo_dir = _setup_repo(repo_ref=repo_ref, repo_url="https://github.com/karpathy/nanochat.git")

    _ensure_uv_env_has_cuda_bits(repo_dir)

    # Find base_train entrypoint (you already confirmed it exists in your run logs)
    base_train = _find_file("base_train.py", repo_dir)
    if not base_train:
        raise FileNotFoundError("Could not find scripts/base_train.py")

    rel = base_train.relative_to(repo_dir)

    # 10-step smoke test; this one saves frequently (every 5) just to prove checkpointing works.
    _uv_run(
        repo_dir,
        f"python {rel} config/{model}.py "
        "--max_iters=10 --log_interval=1 "
        "--eval_interval=5 --always_save_checkpoint=True"
    )

    vol.commit()
    return "Smoke test complete."

@app.local_entrypoint()
def main(task: str = "test", repo_ref: str = "master", model: str = "d12"):
    if task == "run":
        print(run_speedrun.remote(repo_ref=repo_ref, model=model))
    else:
        print(smoke_test_10_steps.remote(repo_ref=repo_ref, model=model))
