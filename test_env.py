"""
test_env.py
-----------
Pre-flight checks before submitting a training job.
Run this interactively on the allocated node:

    python test_env.py
    python test_env.py --wandb-project plankton_2d   # also tests W&B
"""

import argparse
import importlib
import os
import sys


PASS = "  [PASS]"
FAIL = "  [FAIL]"
WARN = "  [WARN]"

failures = []


def check(label, ok, detail="", warn=False):
    tag = WARN if (not ok and warn) else (PASS if ok else FAIL)
    print(f"{tag}  {label}" + (f"  —  {detail}" if detail else ""))
    if not ok and not warn:
        failures.append(label)


# ---------------------------------------------------------------------------
# 1. Python packages
# ---------------------------------------------------------------------------
print("\n── Packages ──────────────────────────────────────────────────────────")

REQUIRED = [
    "torch",
    "torchvision",
    "zarr",
    "tifffile",
    "wandb",
    "segmentation_models_pytorch",
    "scipy",
    "skimage",
    "tqdm",
    "numpy",
]

for pkg in REQUIRED:
    try:
        mod = importlib.import_module(pkg)
        version = getattr(mod, "__version__", "?")
        check(pkg, True, version)
    except ImportError as e:
        check(pkg, False, str(e))


# ---------------------------------------------------------------------------
# 2. SLURM environment
# ---------------------------------------------------------------------------
print("\n── SLURM ─────────────────────────────────────────────────────────────")

job_id = os.environ.get("SLURM_JOB_ID")
check("SLURM_JOB_ID set", bool(job_id), job_id or "not set — are you inside an allocation?", warn=not bool(job_id))

nodelist = os.environ.get("SLURM_NODELIST", "not set")
check("SLURM_NODELIST", bool(os.environ.get("SLURM_NODELIST")), nodelist, warn=True)

gpus_env = os.environ.get("SLURM_GPUS", os.environ.get("SLURM_GPUS_ON_NODE", "not set"))
check("SLURM_GPUS allocated", gpus_env != "not set", gpus_env, warn=True)


# ---------------------------------------------------------------------------
# 3. CUDA / GPU
# ---------------------------------------------------------------------------
print("\n── GPU ───────────────────────────────────────────────────────────────")

try:
    import torch

    cuda_ok = torch.cuda.is_available()
    check("CUDA available", cuda_ok)

    if cuda_ok:
        n_gpus = torch.cuda.device_count()
        check("GPU count", n_gpus >= 1, str(n_gpus))

        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory / 1024**3
            name = props.name
            is_h200 = "H200" in name
            check(
                f"GPU {i}: {name}",
                True,
                f"{mem_gb:.1f} GB" + (" (H200 confirmed)" if is_h200 else " (not H200 — check --constraint)"),
            )

        # Quick forward pass
        try:
            x = torch.zeros(1, 1, 64, 64, device="cuda")
            y = x + 1
            del x, y
            torch.cuda.empty_cache()
            check("CUDA tensor op", True)
        except Exception as e:
            check("CUDA tensor op", False, str(e))

        # fp16
        try:
            x = torch.zeros(1, 1, 64, 64, device="cuda", dtype=torch.float16)
            y = x + 1
            del x, y
            torch.cuda.empty_cache()
            check("fp16 support", True)
        except Exception as e:
            check("fp16 support", False, str(e))

except Exception as e:
    check("torch import", False, str(e))


# ---------------------------------------------------------------------------
# 4. Model instantiation
# ---------------------------------------------------------------------------
print("\n── Model ─────────────────────────────────────────────────────────────")

try:
    from model import build_model
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        model, model_ema, num_classes = build_model(
            project_path = Path(tmp),
            n_channels   = 3,
            architecture = "U-Net",
            encoder      = "resnet34",
            pretrained   = False,
            reset        = True,
        )
    check("build_model (resnet34 / 3ch)", True, f"num_classes={num_classes}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device).eval()
        # Try fp16 first (required for H200 inference path); P40/Pascal will fall back to fp32
        for dtype, label in [(torch.float16, "fp16"), (torch.float32, "fp32")]:
            try:
                m = model.half() if dtype == torch.float16 else model.float()
                x = torch.zeros(2, 3, 256, 256, device=device, dtype=dtype)
                with torch.inference_mode():
                    out = m(x)
                note = "" if dtype == torch.float16 else " (fp16 unsupported on this GPU — H200 will use fp16)"
                check("model forward pass on GPU", True, f"{label}, output {tuple(out.shape)}{note}",
                      warn=(dtype == torch.float32))
                del x, out
                break
            except Exception:
                if dtype == torch.float32:
                    check("model forward pass on GPU", False, "failed in both fp16 and fp32")
                torch.cuda.empty_cache()
        del model, model_ema
        torch.cuda.empty_cache()

except Exception as e:
    check("model forward pass on GPU", False, str(e))


# ---------------------------------------------------------------------------
# 5. Weights & Biases
# ---------------------------------------------------------------------------
print("\n── Weights & Biases ──────────────────────────────────────────────────")

try:
    import wandb

    api_key = os.environ.get("WANDB_API_KEY", "")
    logged_in = bool(api_key) or bool(getattr(wandb.api, "api_key", None))
    check("W&B authenticated", logged_in,
          "via env var" if api_key else ("via wandb login" if logged_in else
          "not authenticated — set WANDB_API_KEY in job script or run `wandb login`"))

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--wandb-project", default=None)
    args, _ = parser.parse_known_args()

    if args.wandb_project:
        try:
            run = wandb.init(
                project = args.wandb_project,
                name    = "env-test",
                config  = {"test": True},
                reinit  = "finish_previous",
            )
            wandb.log({"test_metric": 1.0})
            run.finish()
            check("W&B init + log", True, f"project={args.wandb_project}")
        except Exception as e:
            check("W&B init + log", False, str(e))
    else:
        check("W&B live test", True, "skipped (pass --wandb-project <name> to test)", warn=True)

except ImportError as e:
    check("wandb import", False, str(e))


# ---------------------------------------------------------------------------
# 6. Scratch paths
# ---------------------------------------------------------------------------
print("\n── Paths ─────────────────────────────────────────────────────────────")

scratch_patches = "/scratch/asvetlove/ML_training_patches"
scratch_project = "/scratch/asvetlove/project_2d"

check("patches dir exists",  os.path.isdir(scratch_patches),  scratch_patches, warn=True)
check("project dir writable",
      os.access(os.path.dirname(scratch_project) or ".", os.W_OK),
      scratch_project)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n── Summary ───────────────────────────────────────────────────────────")
if failures:
    print(f"  {len(failures)} check(s) FAILED: {', '.join(failures)}")
    sys.exit(1)
else:
    print("  All checks passed. Ready to train.")
