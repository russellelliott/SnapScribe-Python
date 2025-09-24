#!/usr/bin/env python3

import cv2
import numpy as np
import os
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import asyncio
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Suppress OpenCV warnings
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
cv2.setLogLevel(1)

try:
    from PIL import Image
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_SUPPORT = True
except ImportError:
    HEIC_SUPPORT = False
    print("Warning: pillow-heif not installed. HEIC support disabled.")

# ------------------ QR CODE DETECTOR ------------------ #

class QRCodeDetector:
    """Detects QR codes in images."""

    def __init__(self):
        self.qr_detector = cv2.QRCodeDetector()

    def load_heic_image(self, image_path: str) -> Optional[np.ndarray]:
        if not HEIC_SUPPORT:
            return None
        try:
            pil_image = Image.open(image_path)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Error loading HEIC image: {e}")
            return None

    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        if not os.path.exists(image_path):
            return None
        ext = Path(image_path).suffix.lower()
        if ext in ['.heic', '.heif']:
            return self.load_heic_image(image_path)
        else:
            return cv2.imread(image_path)

    def _apply_preprocessing_technique(self, technique_name: str, gray: np.ndarray) -> np.ndarray:
        if technique_name == "CLAHE":
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            return clahe.apply(gray)
        elif technique_name == "Adaptive":
            return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
        elif technique_name == "Otsu":
            _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return otsu_thresh
        elif technique_name == "OtsuInv":
            _, otsu_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            return otsu_inv
        elif technique_name == "HighContrast":
            return cv2.convertScaleAbs(gray, alpha=2.0, beta=50)
        elif technique_name == "GammaDark":
            return cv2.LUT(gray, np.array([((i / 255.0) ** (1.0 / 0.5)) * 255 for i in range(256)]).astype("uint8"))
        elif technique_name == "GammaBright":
            return cv2.LUT(gray, np.array([((i / 255.0) ** (1.0 / 1.5)) * 255 for i in range(256)]).astype("uint8"))
        elif technique_name == "Bilateral":
            return cv2.bilateralFilter(gray, 9, 75, 75)
        return gray

    def detect_qr_codes(self, image: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        qr_codes = []
        try:
            # Multi-detection
            retval, decoded_info, points, _ = self.qr_detector.detectAndDecodeMulti(image)
            if retval and decoded_info:
                for i, info in enumerate(decoded_info):
                    if info:
                        qr_codes.append((info, points[i]))

            # Single detection if none found
            if not qr_codes:
                data, points, _ = self.qr_detector.detectAndDecode(image)
                if data:
                    qr_codes.append((data, points))

            # Preprocessing techniques
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            found_texts = set([qr[0] for qr in qr_codes])
            techniques = ["CLAHE", "Adaptive", "Otsu", "OtsuInv", "HighContrast",
                          "GammaDark", "GammaBright", "Bilateral"]

            for tech in techniques:
                processed_img = self._apply_preprocessing_technique(tech, gray)
                try:
                    retval, decoded_info, points, _ = self.qr_detector.detectAndDecodeMulti(processed_img)
                    if retval and decoded_info:
                        for i, info in enumerate(decoded_info):
                            if info and info not in found_texts:
                                found_texts.add(info)
                                qr_codes.append((info, points[i]))
                except:
                    continue
        except Exception as e:
            pass
        return qr_codes

    def process_single_file(self, file_path: str) -> Dict[str, Any]:
        filename = os.path.basename(file_path)
        try:
            image = self.load_image(file_path)
            if image is None:
                return {"filename": filename, "success": False,
                        "error": "Failed to load image", "qr_codes": []}

            qr_codes = self.detect_qr_codes(image)
            qr_texts = [qr[0] for qr in qr_codes]

            return {"filename": filename, "success": True, "qr_codes": qr_texts,
                    "count": len(qr_texts)}

        except Exception as e:
            return {"filename": filename, "success": False, "error": str(e), "qr_codes": []}

# ------------------ WORKER FUNCTION ------------------ #

def process_file_worker(file_path: str) -> Dict[str, Any]:
    """Top-level worker function for ProcessPoolExecutor."""
    detector = QRCodeDetector()  # create detector inside process
    return detector.process_single_file(file_path)

# ------------------ FASTAPI ------------------ #

app = FastAPI(title="QR Code Detector API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ------------------ ENDPOINTS ------------------ #

@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    """Serve the frontend HTML interface."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>QR Code Detector</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                text-align: center;
                margin-bottom: 30px;
            }
            .upload-area {
                border: 2px dashed #ccc;
                border-radius: 10px;
                padding: 40px;
                text-align: center;
                margin: 20px 0;
                transition: border-color 0.3s;
                background-color: #fafafa;
            }
            .upload-area:hover {
                border-color: #007bff;
                background-color: #f8f9ff;
            }
            .upload-area.dragover {
                border-color: #007bff;
                background-color: #e3f2fd;
            }
            #file-input {
                display: none;
            }
            .upload-button {
                background: #007bff;
                color: white;
                padding: 12px 24px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                margin: 10px;
                transition: background-color 0.3s;
            }
            .upload-button:hover {
                background: #0056b3;
            }
            .upload-button:disabled {
                background: #ccc;
                cursor: not-allowed;
            }
            #results {
                margin-top: 30px;
                padding: 20px;
                border-radius: 5px;
                display: none;
            }
            .success {
                background-color: #d4edda;
                border: 1px solid #c3e6cb;
                color: #155724;
            }
            .error {
                background-color: #f8d7da;
                border: 1px solid #f5c6cb;
                color: #721c24;
            }
            .file-result {
                background: #f8f9fa;
                padding: 15px;
                margin: 10px 0;
                border-radius: 5px;
                border-left: 4px solid #007bff;
            }
            .qr-code {
                background: white;
                padding: 8px;
                margin: 5px 0;
                border-radius: 3px;
                border: 1px solid #dee2e6;
                font-family: monospace;
                word-break: break-all;
            }
            .loading {
                display: none;
                text-align: center;
                margin: 20px 0;
            }
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #007bff;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .stats {
                background: #e9ecef;
                padding: 15px;
                border-radius: 5px;
                margin: 20px 0;
                text-align: center;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 QR Code Detector</h1>

            <div class="upload-area" id="upload-area">
                <p>Drag and drop image files here, or click to select</p>
                <input type="file" id="file-input" multiple accept="image/*">
                <br>
                <button class="upload-button" id="upload-button">Choose Files</button>
            </div>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Processing images... This may take a few seconds.</p>
            </div>

            <div id="results"></div>
        </div>

        <script>
            const uploadArea = document.getElementById('upload-area');
            const fileInput = document.getElementById('file-input');
            const uploadButton = document.getElementById('upload-button');
            const loading = document.getElementById('loading');
            const results = document.getElementById('results');

            let selectedFiles = [];

            // Handle file selection
            uploadButton.addEventListener('click', () => {
                fileInput.click();
            });

            fileInput.addEventListener('change', (e) => {
                selectedFiles = Array.from(e.target.files);
                updateUploadArea();
            });

            // Drag and drop functionality
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });

            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('dragover');
            });

            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                selectedFiles = Array.from(e.dataTransfer.files);
                updateUploadArea();
            });

            function updateUploadArea() {
                if (selectedFiles.length > 0) {
                    const fileNames = selectedFiles.map(f => f.name).join(', ');
                    uploadArea.innerHTML = `
                        <p><strong>${selectedFiles.length} file(s) selected:</strong></p>
                        <p>${fileNames}</p>
                        <button class="upload-button" id="process-button">Process Images</button>
                        <button class="upload-button" onclick="clearFiles()" style="background: #6c757d;">Clear</button>
                    `;

                    document.getElementById('process-button').addEventListener('click', processFiles);
                }
            }

            function clearFiles() {
                selectedFiles = [];
                fileInput.value = '';
                uploadArea.innerHTML = `
                    <p>Drag and drop image files here, or click to select</p>
                    <input type="file" id="file-input" multiple accept="image/*" style="display: none;">
                    <br>
                    <button class="upload-button" id="upload-button">Choose Files</button>
                `;
                document.getElementById('upload-button').addEventListener('click', () => fileInput.click());
                fileInput.addEventListener('change', (e) => {
                    selectedFiles = Array.from(e.target.files);
                    updateUploadArea();
                });
            }

            async function processFiles() {
                if (selectedFiles.length === 0) return;

                loading.style.display = 'block';
                results.style.display = 'none';
                results.innerHTML = '';

                const formData = new FormData();
                selectedFiles.forEach(file => {
                    formData.append('files', file);
                });

                try {
                    const response = await fetch('/api/detect-qr', {
                        method: 'POST',
                        body: formData
                    });

                    const data = await response.json();

                    if (response.ok) {
                        displayResults(data);
                    } else {
                        showError(data.detail || 'An error occurred');
                    }
                } catch (error) {
                    showError('Network error: ' + error.message);
                } finally {
                    loading.style.display = 'none';
                }
            }

            function displayResults(data) {
                results.style.display = 'block';
                results.className = 'success';

                let totalQRCodes = 0;
                let processedFiles = 0;

                let html = '<h3>✅ Processing Complete!</h3>';

                if (data.results && data.results.length > 0) {
                    data.results.forEach(result => {
                        if (result.success) {
                            processedFiles++;
                            totalQRCodes += result.count;

                            html += `
                                <div class="file-result">
                                    <h4>📁 ${result.filename}</h4>
                                    <p><strong>QR Codes Found: ${result.count}</strong></p>
                            `;

                            if (result.qr_codes && result.qr_codes.length > 0) {
                                result.qr_codes.forEach((qr, index) => {
                                    html += `<div class="qr-code">${index + 1}. ${qr}</div>`;
                                });
                            }

                            html += '</div>';
                        } else {
                            html += `
                                <div class="file-result" style="border-left-color: #dc3545;">
                                    <h4>❌ ${result.filename}</h4>
                                    <p>Error: ${result.error}</p>
                                </div>
                            `;
                        }
                    });
                }

                html += `
                    <div class="stats">
                        <h4>📊 Summary</h4>
                        <p>Files Processed: ${processedFiles} | Total QR Codes Found: ${totalQRCodes}</p>
                        <p>Processing Time: ${data.processing_time ? data.processing_time.toFixed(2) + 's' : 'N/A'}</p>
                    </div>
                `;

                results.innerHTML = html;
            }

            function showError(message) {
                results.style.display = 'block';
                results.className = 'error';
                results.innerHTML = `<h3>❌ Error</h3><p>${message}</p>`;
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/api/detect-qr")
async def detect_qr_codes_endpoint(files: List[UploadFile] = File(...)):
    import time
    start_time = time.time()
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    with tempfile.TemporaryDirectory() as temp_dir:
        file_paths = []
        for file in files:
            if not file.filename:
                continue
            ext = Path(file.filename).suffix.lower()
            if ext not in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.heic', '.heif'}:
                continue
            fp = os.path.join(temp_dir, file.filename)
            with open(fp, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            file_paths.append(fp)

        if not file_paths:
            raise HTTPException(status_code=400, detail="No valid image files found")

        max_workers = min(len(file_paths), multiprocessing.cpu_count(), 8)
        loop = asyncio.get_event_loop()

        # Process images in parallel using processes
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [loop.run_in_executor(executor, process_file_worker, fp)
                       for fp in file_paths]
            results = await asyncio.gather(*futures, return_exceptions=True)

        processed_results = []
        for res in results:
            if isinstance(res, Exception):
                processed_results.append({"filename": "unknown", "success": False,
                                          "error": str(res), "qr_codes": []})
            else:
                processed_results.append(res)

    processing_time = time.time() - start_time
    return JSONResponse({"success": True, "processing_time": processing_time,
                         "results": processed_results})

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "QR Code Detector API"}

# ------------------ MAIN ------------------ #

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="QR Code Detector FastAPI Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    uvicorn.run("main:app", host=args.host, port=args.port, reload=args.reload)
