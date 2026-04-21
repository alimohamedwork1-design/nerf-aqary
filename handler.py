import os
import uuid
import shutil
import subprocess
import requests
from pathlib import Path
import runpod


GS_DIR = Path("/workspace/gaussian-splatting")

SUPABASE_URL = os.environ["SUPABASE_URL"].rstrip("/")
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
BUCKET = "models"

HEADERS = {
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "apikey": SUPABASE_KEY
}

def update_db(tour_id, status, progress=0, model_url=None):
    headers = HEADERS.copy()
    headers["Content-Type"] = "application/json"

    payload = {
        "status": status,
        "progress": progress
    }

    if model_url:
        payload["ply_url"] = model_url

    requests.patch(
        f"{SUPABASE_URL}/rest/v1/virtual_tours?id=eq.{tour_id}",
        headers=headers,
        json=payload
    )

def upload_file(path, tour_id):
    file_name = f"{tour_id}.ply"

    headers = HEADERS.copy()
    headers["Content-Type"] = "application/octet-stream"
    headers["x-upsert"] = "true"

    with open(path, "rb") as f:
        r = requests.post(
            f"{SUPABASE_URL}/storage/v1/object/{BUCKET}/{file_name}",
            headers=headers,
            data=f
        )

    if r.status_code >= 300:
        raise Exception(r.text)

    return f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET}/{file_name}"

def run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(result.stderr)
    return result.stdout

def handler(job):
    data = job["input"]

    tour_id = data["tour_id"]
    images = data["image_urls"]
    iterations = str(data.get("iterations", 2000))

    work = Path(f"/tmp/{tour_id}")
    imgs = work / "images"
    out = work / "output"

    try:
        imgs.mkdir(parents=True)

        update_db(tour_id, "downloading", 10)

        for i, url in enumerate(images):
            img = requests.get(url).content
            (imgs / f"{i}.jpg").write_bytes(img)

        update_db(tour_id, "processing", 30)

        run([
            "python3",
            f"{GS_DIR}/convert.py",
            "-s",
            str(work),
            "--no_gpu"
        ])

        update_db(tour_id, "training", 60)

        run([
            "python3",
            f"{GS_DIR}/train.py",
            "-s",
            str(work),
            "-m",
            str(out),
            "--iterations",
            iterations
        ])

        ply = list(out.rglob("point_cloud.ply"))[-1]

        update_db(tour_id, "uploading", 90)

        url = upload_file(ply, tour_id)

        update_db(tour_id, "completed", 100, url)

        return {
            "status": "done",
            "url": url
        }

    except Exception as e:
        update_db(tour_id, "failed", 0)
        return {"error": str(e)}

    finally:
        shutil.rmtree(work, ignore_errors=True)

runpod.serverless.start({"handler": handler})
