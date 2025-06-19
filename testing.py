# # testing.py
# import requests
# import time
# import cv2
# import face_recognition

# SERVER_URL = "http://localhost:8000/search"
# TEST_IMAGE = r"enhanced_faces_dataset\0.jpg"

# def verify_test_image():
#     print("\nVerifying test image...")
#     img = cv2.imread(TEST_IMAGE)
#     if img is None:
#         print("ERROR: Could not read image file")
#         return
#     print(f"Image loaded (shape: {img.shape})")
#     rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     locs = face_recognition.face_locations(rgb)
#     print(f"Detected {len(locs)} face(s)")
#     if not locs:
#         print("WARNING: No faces detected!")

# def send_request():
#     try:
#         with open(TEST_IMAGE, 'rb') as f:
#             r = requests.post(
#                 SERVER_URL,
#                 files={'file': (TEST_IMAGE, f, 'image/jpeg')},
#                 timeout=10
#             )
#         latency = r.elapsed.total_seconds()
#         # Try JSON, else raw text
#         try:
#             data = r.json()
#         except ValueError:
#             data = r.text
#         return {
#             "status": r.status_code,
#             "success": r.status_code == 200,
#             "latency": latency,
#             "data": data
#         }
#     except Exception as e:
#         return {"status": None, "success": False, "error": str(e)}

# def run_tests():
#     verify_test_image()
#     print("\nRunning single test request...")
#     res = send_request()
#     if res["success"]:
#         print("✅ Success:")
#         print(res["data"])
#         print(f"Latency: {res['latency']:.3f}s")
#     else:
#         print("❌ Test failed!")
#         if res.get("error"):
#             print("Error:", res["error"])
#         else:
#             print("Status code:", res["status"])
#             print("Response body:", res["data"])

# if __name__ == "__main__":
#     run_tests()


# import requests
# import cv2
# import time
# from concurrent.futures import ThreadPoolExecutor
# import matplotlib.pyplot as plt

# # Configuration
# SERVER_URL = "http://localhost:8000/search"
# TEST_IMAGE = "enhanced_faces_dataset/1.jpg"
# DATASET_PATH = "enhanced_faces_dataset"

# # 1. Single request and display query + top-5 matches
# response = requests.post(
#     SERVER_URL,
#     files={'file': (TEST_IMAGE, open(TEST_IMAGE, 'rb'), 'image/jpeg')}
# )
# response.raise_for_status()
# data = response.json()

# # Load and prepare images
# query_img = cv2.cvtColor(cv2.imread(TEST_IMAGE), cv2.COLOR_BGR2RGB)
# matches = []
# for match in data['results']:
#     img_path = f"{DATASET_PATH}/{match['filename']}"
#     img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
#     matches.append((img, match['similarity']))

# # Plot query + matches
# fig, axes = plt.subplots(1, 6, figsize=(15, 5))
# axes[0].imshow(query_img)
# axes[0].set_title("Query")
# axes[0].axis('off')

# for i, (img, sim) in enumerate(matches):
#     axes[i+1].imshow(img)
#     axes[i+1].set_title(f"{sim:.2f}")
#     axes[i+1].axis('off')

# plt.show()

# # 2. Throughput test: 20 requests per second, record latencies
# latencies = []
# def send_request():
#     start = time.time()
#     with open(TEST_IMAGE, 'rb') as f:
#         r = requests.post(
#             SERVER_URL,
#             files={'file': (TEST_IMAGE, f, 'image/jpeg')}
#         )
#     latencies.append((time.time() - start) * 1000)

# # Schedule 100 requests at ~20 RPS
# with ThreadPoolExecutor(max_workers=20) as executor:
#     for _ in range(100):
#         executor.submit(send_request)
#         time.sleep(1/20)

# # Print average latency
# avg_latency = sum(latencies) / len(latencies)
# print(f"Average latency @20 RPS: {avg_latency:.2f} ms over {len(latencies)} requests")

# # 3. Plot latency distribution
# plt.figure(figsize=(8, 4))
# plt.hist(latencies, bins=20)
# plt.xlabel("Latency (ms)")
# plt.ylabel("Count")
# plt.title("Latency Distribution @20 RPS")
# plt.show()
import glob
import os
import time
import requests
import cv2
import matplotlib.pyplot as plt
import numpy as np              # ← for isinstance check

# ── CONFIG ───────────────────────────────────────────────────
SERVER_URL = "http://localhost:8000/search_batch"
IMAGE_DIR  = "enhanced_faces_dataset"
MAX_IMAGES = 30
# ─────────────────────────────────────────────────────────────

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
