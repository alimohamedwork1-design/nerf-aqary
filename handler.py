## “””
ARqary NeRF/3DGS RunPod Serverless Handler

Input:
{
“input”: {
“job_id”: “uuid”,
“image_urls”: [“https://r2…/img1.jpg”, …],  // 10+ images from R2
“bucket”: “arqary-storage”,
“output_key”: “tours/{job_id}/output.ply”
}
}

Output:
{
“status”: “completed” | “failed”,
“ply_url”: “https://r2…/output.ply”,
“error”: “…” // only on failure
}
“””

import runpod
import os
import json
import subprocess
import shutil
import urllib.request
import boto3
from pathlib import Path

# ── R2 Config (set as RunPod secrets) ──────────────────────────────────────────

R2_ENDPOINT   = os.environ.get(“R2_ENDPOINT”)        # https://<account>.r2.cloudflarestorage.com
R2_ACCESS_KEY = os.environ.get(“R2_ACCESS_KEY”)
R2_SECRET_KEY = os.environ.get(“R2_SECRET_KEY”)
R2_BUCKET     = os.environ.get(“R2_BUCKET”, “arqary-storage”)
R2_PUBLIC_URL = os.environ.get(“R2_PUBLIC_URL”)      # https://pub-xxx.r2.dev  (public bucket URL)

# ── Paths ───────────────────────────────────────────────────────────────────────

WORKSPACE       = Path(”/tmp/arqary_job”)
IMAGES_DIR      = WORKSPACE / “images”
COLMAP_DIR      = WORKSPACE / “colmap”
SPARSE_DIR      = COLMAP_DIR / “sparse”
OUTPUT_DIR      = WORKSPACE / “output”
GAUSSIAN_SCRIPT = Path(”/workspace/gaussian-splatting/train.py”)

def get_r2_client():
return boto3.client(
“s3”,
endpoint_url=R2_ENDPOINT,
aws_access_key_id=R2_ACCESS_KEY,
aws_secret_access_key=R2_SECRET_KEY,
region_name=“auto”,
)

def download_images(image_urls: list[str]) -> int:
“”“Download images from R2 public URLs to IMAGES_DIR.”””
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
count = 0
for i, url in enumerate(image_urls):
ext = url.split(”.”)[-1].split(”?”)[0].lower()
if ext not in (“jpg”, “jpeg”, “png”, “webp”):
ext = “jpg”
dest = IMAGES_DIR / f”image_{i:04d}.{ext}”
print(f”[download] {url} -> {dest.name}”)
urllib.request.urlretrieve(url, dest)
count += 1
return count

def run_colmap() -> bool:
“”“Run COLMAP feature extraction + matching + mapping (CPU SiftExtraction).”””
SPARSE_DIR.mkdir(parents=True, exist_ok=True)
db = COLMAP_DIR / “database.db”

```
steps = [
    # Feature extraction
    [
        "colmap", "feature_extractor",
        "--database_path", str(db),
        "--image_path", str(IMAGES_DIR),
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.use_gpu", "0",   # CPU mode — avoids SIGABRT on some GPUs
    ],
    # Exhaustive matching
    [
        "colmap", "exhaustive_matcher",
        "--database_path", str(db),
        "--SiftMatching.use_gpu", "0",
    ],
    # Mapping
    [
        "colmap", "mapper",
        "--database_path", str(db),
        "--image_path", str(IMAGES_DIR),
        "--output_path", str(SPARSE_DIR),
    ],
    # Convert to TXT for gaussian-splatting
    [
        "colmap", "model_converter",
        "--input_path", str(SPARSE_DIR / "0"),
        "--output_path", str(SPARSE_DIR / "0"),
        "--output_type", "TXT",
    ],
]

for cmd in steps:
    print(f"[colmap] {' '.join(cmd[:2])}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[colmap ERROR]\n{result.stderr}")
        return False
    print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)

return True
```

def run_gaussian_splatting(job_id: str) -> Path | None:
“”“Train 3DGS and return path to output .ply file.”””
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

```
cmd = [
    "python", str(GAUSSIAN_SCRIPT),
    "--source_path", str(COLMAP_DIR),
    "--model_path", str(OUTPUT_DIR),
    "--iterations", "3000",       # Fast training for serverless (full = 30000)
    "--sh_degree", "1",
]

print(f"[3dgs] Starting training (3000 iterations)...")
result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min max

if result.returncode != 0:
    print(f"[3dgs ERROR]\n{result.stderr[-1000:]}")
    return None

# Find the output .ply (point_cloud/iteration_3000/point_cloud.ply)
ply_candidates = list(OUTPUT_DIR.rglob("point_cloud.ply"))
if not ply_candidates:
    print("[3dgs ERROR] No .ply file found after training")
    return None

# Pick the latest iteration
ply_path = sorted(ply_candidates)[-1]
print(f"[3dgs] Output: {ply_path}")
return ply_path
```

def upload_to_r2(local_path: Path, output_key: str) -> str:
“”“Upload .ply to R2 and return public URL.”””
client = get_r2_client()
print(f”[r2] Uploading {local_path.name} -> {output_key}”)
client.upload_file(
str(local_path),
R2_BUCKET,
output_key,
ExtraArgs={“ContentType”: “application/octet-stream”},
)
public_url = f”{R2_PUBLIC_URL.rstrip(’/’)}/{output_key}”
print(f”[r2] Done: {public_url}”)
return public_url

def cleanup():
if WORKSPACE.exists():
shutil.rmtree(WORKSPACE)
print(”[cleanup] Done”)

# ── RunPod Handler ──────────────────────────────────────────────────────────────

def handler(event):
job_input = event.get(“input”, {})

```
job_id     = job_input.get("job_id", "unknown")
image_urls = job_input.get("image_urls", [])
output_key = job_input.get("output_key", f"tours/{job_id}/output.ply")

print(f"[job] {job_id} | {len(image_urls)} images")

# Validate
if len(image_urls) < 5:
    return {"status": "failed", "error": "Need at least 5 images for reconstruction"}

try:
    # 1. Download images
    count = download_images(image_urls)
    print(f"[job] Downloaded {count} images")

    # 2. COLMAP
    if not run_colmap():
        return {"status": "failed", "error": "COLMAP failed — check image overlap/quality"}

    # 3. Gaussian Splatting
    ply_path = run_gaussian_splatting(job_id)
    if ply_path is None:
        return {"status": "failed", "error": "3DGS training failed"}

    # 4. Upload to R2
    ply_url = upload_to_r2(ply_path, output_key)

    return {
        "status": "completed",
        "ply_url": ply_url,
        "job_id": job_id,
    }

except Exception as e:
    print(f"[job ERROR] {e}")
    return {"status": "failed", "error": str(e)}

finally:
    cleanup()
```

if **name** == “**main**”:
runpod.serverless.start({“handler”: handler})
