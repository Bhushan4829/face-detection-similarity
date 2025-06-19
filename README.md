# Face Detection Similarity

A high-performance face similarity search system that processes 20+ requests per second on CPU-only hardware with sub-2-second latency.

## 🚀 Features

- **High Performance**: Process 20+ requests per second on CPU-only hardware
- **Multi-Input Support**: Handle both image and video inputs
- **Fast Similarity Search**: Return top 5 most similar faces from 1,000+ face database
- **Low Latency**: Maintains sub-2-second response time per request of image
- **Web Interface**: User-friendly web UI for easy interaction
- **Batch Processing**: Support for processing multiple queries simultaneously

## 📁 Project Structure
face-detection-similarity/
├── database.py          # Face database creation and indexing
├── main.py             # FastAPI server implementation
├── download_models.py  # Model download utility
├── requirements.txt    # Python dependencies
├── static/            # Web interface files
│   ├── index.html
│   ├── style.css
│   └── script.js
└── README.md

## 🔧 Key Components

### 1. Face Database Creation (`database.py`)

**Functionality:**
- Processes 1,000 unique faces from the FFHQ dataset
- Extracts facial encodings using face_recognition library
- Creates optimized FAISS index for fast similarity search
- Implements clustering for enhanced performance

**Optimizations:**
- **IVF Indexing**: Uses Inverted File indexing with 100 Voronoi cells
- **Vector Normalization**: Normalizes vectors for cosine similarity
- **Clustering**: Implements MiniBatchKMeans clustering
- **Memory Efficiency**: Stores data in optimized formats

### 2. API Server (`main.py`)

**Features:**
- **FastAPI Backend**: Async processing with high throughput
- **Dual Endpoints**:
  - `/search` - Single image queries
  - `/search_batch` - Batch processing
- **Video Support**: Built-in video frame extraction
- **Static Serving**: Serves web UI files

**Performance Optimizations:**
- **Thread Pool**: Dedicated pool for CPU-bound tasks
- **YuNet Face Detector**: Ultra-fast detection (5-10ms per face)
- **OpenCV DNN**: Efficient face encoding
- **FAISS Integration**: Tuned nprobe settings for optimal speed
- **Batch Processing**: Efficient handling of multiple queries

### 3. Web Interface (`static/`)

**Components:**
- **Video Frame Extractor**: Adjustable FPS extraction
- **Real-time Results**: Live display of similarity matches
- **Side-by-side Comparison**: Query vs. match visualization
- **Progress Indicators**: Real-time processing feedback

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Step 1: Install Dependencies
```
pip install -r requirements.txt
``` 
### Step 2: Download Required Models
``` 
python download_models.py
```
### Step 3: Create Face Database
``` 
python database.py
```
Note: This process may take several minutes as it processes 1,000 faces and creates the search index.
### Step 4: Start the Server
``` 
python main.py
```
### Step 5: Access Web Interface
Open your browser and navigate to:
``` 
http://localhost:8000
```

📋 Requirements
```
txtfastapi>=0.68.0
uvicorn>=0.15.0
face-recognition>=1.3.0
opencv-python>=4.5.0
faiss-cpu>=1.7.0
numpy>=1.21.0
scikit-learn>=1.0.0
Pillow>=8.3.0
python-multipart>=0.0.5
```
🚀 API Usage
Single Image Search
```
curl -X POST "http://localhost:8000/search" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_image.jpg"
Batch Search
curl -X POST "http://localhost:8000/search_batch" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "files=@image1.jpg" \
     -F "files=@image2.jpg"
⚡ Performance Specifications 
```
Throughput: 20+ requests per second
Latency: < 2 seconds per request
Database Size: 1,000 faces
Hardware: CPU-only (no GPU required)
Face Detection: 5-10ms per face
Results: Top 5 most similar faces

🔍 Technical Details
Face Detection & Recognition

Detector: YuNet (OpenCV DNN)
Encoder: face_recognition library
Similarity Metric: Cosine similarity

Search Index

Engine: FAISS (Facebook AI Similarity Search)
Index Type: IVF (Inverted File)
Clustering: 100 Voronoi cells
Vector Dimension: 128 (face encoding size)

Optimization Techniques

Vector Normalization: For consistent cosine similarity
Memory Mapping: Efficient data loading
Async Processing: Non-blocking request handling
Thread Pooling: Parallel processing of CPU-bound tasks

