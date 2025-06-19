# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# import face_recognition
# import numpy as np
# import pickle
# import faiss
# import io
# import traceback
# from typing import List

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# def load_database():
#     print("Loading FAISS database...")
#     try:
#         with open('face_db_faiss.pkl', 'rb') as f:
#             db = pickle.load(f)
        
#         # Verify database structure
#         required_keys = {'index', 'filenames'}
#         if not all(k in db for k in required_keys):
#             raise ValueError("Database missing required keys")
            
#         return db
#     except Exception as e:
#         print(f"Database loading failed: {str(e)}")
#         print(traceback.format_exc())
#         raise

# try:
#     db = load_database()
#     index = db['index']
#     filenames = db['filenames']
#     index.nprobe = 5
# except Exception as e:
#     print(f"Failed to initialize: {str(e)}")
#     exit(1)

# def search_similar_faces(query_encoding: np.ndarray, k: int = 5) -> List[str]:
#     try:
#         query_encoding = query_encoding.astype('float32').reshape(1, -1)
#         faiss.normalize_L2(query_encoding)
#         distances, indices = index.search(query_encoding, k)
#         return [filenames[i] for i in indices[0]]
#     except Exception as e:
#         print(f"Search failed: {str(e)}")
#         return []

# @app.post("/search")
# async def search_faces(file: UploadFile = File(...)):
#     try:
#         contents = await file.read()
#         img = face_recognition.load_image_file(io.BytesIO(contents))
        
#         # Try multiple face detection methods
#         face_locations = face_recognition.face_locations(img, model="hog") or \
#                        face_recognition.face_locations(img, model="cnn")
        
#         if not face_locations:
#             raise HTTPException(400, "No faces detected (tried both HOG and CNN models)")
            
#         # Get the largest face if multiple detected
#         face_encoding = face_recognition.face_encodings(img, known_face_locations=face_locations)[0]
#         results = search_similar_faces(face_encoding)
        
#         if not results:
#             raise HTTPException(400, "No similar faces found")
            
#         return {"results": results}
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         print(f"Processing error: {str(e)}")
#         print(traceback.format_exc())
#         raise HTTPException(500, f"Processing failed: {str(e)}")

# @app.get("/health")
# def health_check():
#     return {"status": "healthy", "database_loaded": bool(db)}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

# main.py
# main_async.py

# main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
import io, time, pickle, asyncio
import numpy as np
import faiss
import face_recognition
from typing import List, Dict, Any

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load your FAISS DB
with open("face_db_optimized.pkl", "rb") as f:
    db = pickle.load(f)
print("Index type:", type(db["index"]))

if isinstance(db["index"], faiss.IndexIVF):
    db["index"].nprobe = db["config"].get("nprobe", 5)
# ──────────── Utility Functions ────────────

def calculate_box_similarity(b1: Dict[str,int], b2: Dict[str,int]) -> float:
    if not b1 or not b2:
        return 0.0
    w1, h1 = b1["right"]-b1["left"], b1["bottom"]-b1["top"]
    w2, h2 = b2["right"]-b2["left"], b2["bottom"]-b2["top"]
    ar1, ar2 = w1/h1, w2/h2
    ar_sim = 1 - abs(ar1-ar2)/max(ar1,ar2)
    area1, area2 = w1*h1, w2*h2
    area_sim = 1 - abs(area1-area2)/max(area1,area2)
    return (ar_sim + area_sim) / 2

def context_aware_search(
    query_encoding: np.ndarray,
    query_box: Dict[str,int],
    k: int = 5
) -> List[Dict[str,Any]]:
    q = query_encoding.astype("float32").reshape(1, -1)
    faiss.normalize_L2(q)
    distances, indices = db["index"].search(q, k)

    out = []
    for dist, idx in zip(distances[0], indices[0]):
        fn = db["filenames"][idx]
        box = db["face_boxes"].get(fn, {})
        out.append({
            "filename": fn,
            "similarity": float(dist),
            "face_box": box,
            "box_similarity": calculate_box_similarity(query_box, box)
        })
    return out

async def process_upload(file: UploadFile) -> Dict[str,Any]:
    start = time.time()*1000
    res: Dict[str,Any] = {"filename": file.filename}
    try:
        data = await file.read()

        # 1) load image
        img = await run_in_threadpool(face_recognition.load_image_file,
                                      io.BytesIO(data))

        # 2) detect
        locs = await run_in_threadpool(face_recognition.face_locations, img)
        if not locs:
            res["error"] = "no face detected"
            return res
        top, right, bottom, left = locs[0]
        query_box = {"top": top, "right": right, "bottom": bottom, "left": left}

        # 3) encode
        encs = await run_in_threadpool(face_recognition.face_encodings,
                                       img, locs)
        if not encs:
            res["error"] = "encoding failed"
            return res
        query_vec = np.array(encs[0])

        # 4) FAISS search
        matches = await run_in_threadpool(
            context_aware_search,
            query_vec,
            query_box,
            5
        )

        # 5) finalize
        elapsed = time.time()*1000 - start
        res.update({
            "query_box": query_box,
            "results":   matches,
            "latency_ms": round(elapsed, 2)
        })
        return res

    except Exception as e:
        res["error"] = str(e)
        return res

# ──────────── API Endpoints ────────────

@app.post("/search")
async def search_single(file: UploadFile = File(...)):
    res = await process_upload(file)
    if "error" in res:
        raise HTTPException(400, res["error"])
    return res

@app.post("/search_batch")
async def search_batch(files: List[UploadFile] = File(...)):
    tasks = [process_upload(f) for f in files]
    results = await asyncio.gather(*tasks)
    return {"batch_results": results}

@app.get("/health")
def health():
    return {"status": "healthy", "loaded": len(db["filenames"])}

# ─── MOUNT STATIC ───
# 1) Serve the actual image files
app.mount("/images",
          StaticFiles(directory="enhanced_faces_dataset"),
          name="images")

# 2) Then serve your web UI
app.mount("/",
          StaticFiles(directory="static", html=True),
          name="static")


# ──────────── Run with Uvicorn ────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)

