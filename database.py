# prepare_database.py
# import pandas as pd
# import numpy as np
# import pickle

# # Load the CSV with face encodings
# df = pd.read_csv('face_features.csv')

# # Convert string representations back to numpy arrays
# df['face_encoding'] = df['face_encoding'].apply(lambda x: np.array(eval(x)))

# # Create optimized data structure
# face_data = {
#     'filenames': df['filename'].tolist(),
#     'encodings': np.array(df['face_encoding'].tolist())
# }

# # Save for production
# with open('face_db.pkl', 'wb') as f:
#     pickle.dump(face_data, f, protocol=4)

# prepare_database.py
# prepare_database.py
# database.py
import pandas as pd
import numpy as np
import pickle
import faiss
from sklearn.cluster import MiniBatchKMeans

def create_optimized_database(
    csv_path: str = "face_features.csv",
    output_path: str = "face_db_optimized.pkl"
):
    # 1. Load CSV
    df = pd.read_csv(csv_path)
    filenames = df["filename"].tolist()

    # 2. Parse encodings into an N×128 float32 array
    face_encodings = np.array([
        np.array(eval(x)).astype("float32")
        for x in df["face_encoding"]
    ])

    # 3. Build a filename → bounding‐box dict
    face_boxes = {
        row["filename"]: {
            "top":    row["face_top"],
            "right":  row["face_right"],
            "bottom": row["face_bottom"],
            "left":   row["face_left"]
        }
        for _, row in df.iterrows()
    }

    # 4. Normalize vectors and build FAISS IVF‐Flat index
    faiss.normalize_L2(face_encodings)
    d = face_encodings.shape[1]  # 128
    nlist = 100
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    assert face_encodings.shape[0] >= nlist, "need ≥ nlist vectors to train"
    index.train(face_encodings)
    index.add(face_encodings)
    index.make_direct_map()

    # 5. Optional clustering for further speedups or filtering
    n_clusters = min(50, face_encodings.shape[0])
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1000, random_state=42)
    cluster_labels = kmeans.fit_predict(face_encodings)

    # 6. Save everything to pickle
    db = {
        "filenames":     filenames,
        "face_boxes":    face_boxes,
        "face_encodings": face_encodings,
        "index":         index,
        "cluster_model": kmeans,
        "cluster_labels": cluster_labels,
        "config": {
            "nprobe": 5,
            "metric": "cosine"
        }
    }
    with open(output_path, "wb") as f:
        pickle.dump(db, f, protocol=4)

    print(f"✅ Saved {len(filenames)} faces with {n_clusters} clusters → {output_path}")


if __name__ == "__main__":
    create_optimized_database()
