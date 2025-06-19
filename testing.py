import glob
import os
import time
import requests
import cv2
import matplotlib.pyplot as plt
import numpy as np

# CONFIG 
SERVER_URL = "http://localhost:8000/search_batch"
IMAGE_DIR  = "enhanced_faces_dataset"
MAX_IMAGES = 30


def main():
    # 1) pick up to MAX_IMAGES files
    all_paths = glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
    test_paths = all_paths[:MAX_IMAGES]
    if not test_paths:
        print("No images found in", IMAGE_DIR)
        return

    # 2) prepare multipart with only basenames
    files = []
    for full in test_paths:
        base = os.path.basename(full)
        files.append(("files", (base, open(full, "rb"), "image/jpeg")))

    # 3) send batch and time it
    start = time.time()
    resp = requests.post(SERVER_URL, files=files, timeout=60)
    batch_ms = (time.time() - start) * 1000
    resp.raise_for_status()
    batch_results = resp.json().get("batch_results", [])

    print(f"→ Sent {len(test_paths)} images in one batch "
          f"(search_batch) → total latency: {batch_ms:.1f} ms\n")

    # 4) display each query + its top-5 matches
    for entry in batch_results:
        qf = entry["filename"]
        matches = entry.get("results", [])

        query_path = os.path.join(IMAGE_DIR, os.path.basename(qf))
        img_q = cv2.imread(query_path)
        if img_q is None:
            print("⚠️  Could not load query", query_path)
            continue
        img_q = cv2.cvtColor(img_q, cv2.COLOR_BGR2RGB)

        cols = len(matches) + 1
        fig, axes = plt.subplots(1, cols, figsize=(3*cols, 3))
        # MAKE SURE axes IS A LIST
        if not isinstance(axes, (list, tuple, np.ndarray)):
            axes = [axes]

        # Query
        axes[0].imshow(img_q)
        axes[0].set_title(f"Query\n{os.path.basename(qf)}")
        axes[0].axis("off")

        # Matches
        for i, m in enumerate(matches):
            fn, sim = m["filename"], m["similarity"]
            mp = os.path.join(IMAGE_DIR, os.path.basename(fn))
            img_m = cv2.imread(mp)
            if img_m is None:
                print("⚠️  Could not load match", mp)
                continue
            img_m = cv2.cvtColor(img_m, cv2.COLOR_BGR2RGB)

            axes[i+1].imshow(img_m)
            axes[i+1].set_title(f"{os.path.basename(fn)}\n{sim:.2f}")
            axes[i+1].axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
