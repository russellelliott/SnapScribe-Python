#!/usr/bin/env python3

import cv2
import numpy as np
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import argparse
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
import asyncio
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import shutil
import json
from datetime import datetime

# Suppress OpenCV warnings about ECI encoding (Extended Channel Interpretation)
# These warnings appear when QR codes contain non-ASCII characters but are harmless
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

try:
    from PIL import Image
    from pillow_heif import register_heif_opener
    # Register HEIF opener with Pillow
    register_heif_opener()
    HEIC_SUPPORT = True
except ImportError:
    HEIC_SUPPORT = False
    print("Warning: pillow-heif not installed. HEIC support disabled.")
    print("Install with: pip install pillow-heif")

# Additional OpenCV warning suppression
cv2.setLogLevel(1)  # Only show errors


class QRCodeDetector:
    """A class to detect and decode QR codes in images, including HEIC files."""

    def __init__(self):
        """Initialize the QR code detector."""
        self.qr_detector = cv2.QRCodeDetector()

    def load_heic_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load a HEIC image file and convert it to OpenCV format.

        Args:
            image_path (str): Path to the HEIC image file

        Returns:
            Optional[np.ndarray]: OpenCV image array or None if failed
        """
        if not HEIC_SUPPORT:
            print("Error: HEIC support not available. Please install pillow-heif.")
            return None

        try:
            # Open HEIC file with PIL
            pil_image = Image.open(image_path)

            # Convert PIL image to RGB if not already
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')

            # Convert PIL image to OpenCV format (BGR)
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            return cv_image

        except Exception as e:
            print(f"Error loading HEIC image: {e}")
            return None

    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load an image file, supporting both regular formats and HEIC.

        Args:
            image_path (str): Path to the image file

        Returns:
            Optional[np.ndarray]: OpenCV image array or None if failed
        """
        if not os.path.exists(image_path):
            print(f"Error: Image file not found: {image_path}")
            return None

        # Check if it's a HEIC file
        file_extension = Path(image_path).suffix.lower()
        if file_extension in ['.heic', '.heif']:
            return self.load_heic_image(image_path)
        else:
            # Use OpenCV for regular image formats
            try:
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Error: Could not load image: {image_path}")
                return image
            except Exception as e:
                print(f"Error loading image: {e}")
                return None

    def _apply_preprocessing_technique(self, args) -> List[Tuple[str, np.ndarray]]:
        """
        Apply a single preprocessing technique and detect QR codes.
        Helper method for parallel processing.

        Args:
            args: Tuple of (technique_name, processed_img, scale, found_texts)

        Returns:
            List[Tuple[str, np.ndarray]]: List of new QR codes found
        """
        technique_name, processed_img, scale, found_texts = args

        # Apply scale if needed
        if scale != 1.0:
            height, width = processed_img.shape[:2]
            new_width = int(width * scale)
            new_height = int(height * scale)
            scaled_img = cv2.resize(processed_img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        else:
            scaled_img = processed_img

        new_qr_codes = []

        try:
            # Try multi-detection
            retval, decoded_info, points, _ = self.qr_detector.detectAndDecodeMulti(scaled_img)
            if retval and decoded_info:
                for i, info in enumerate(decoded_info):
                    if info and info not in found_texts:  # Avoid duplicates
                        # Scale points back if we resized the image
                        if scale != 1.0:
                            scaled_points = points[i] / scale
                            new_qr_codes.append((info, scaled_points.astype(np.float32)))
                        else:
                            new_qr_codes.append((info, points[i]))

            # Also try single detection
            data, single_points, _ = self.qr_detector.detectAndDecode(scaled_img)
            if data and data not in found_texts:
                if scale != 1.0:
                    scaled_points = single_points / scale
                    new_qr_codes.append((data, scaled_points.astype(np.float32)))
                else:
                    new_qr_codes.append((data, single_points))

        except Exception as e:
            pass  # Continue on error

        return new_qr_codes

    def detect_qr_codes(self, image: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """
        Detect and decode QR codes in an image with parallel preprocessing.

        Args:
            image (np.ndarray): OpenCV image array

        Returns:
            List[Tuple[str, np.ndarray]]: List of (decoded_text, corner_points) tuples
        """
        qr_codes = []

        try:
            # First, try to detect multiple QR codes at once
            retval, decoded_info, points, _ = self.qr_detector.detectAndDecodeMulti(image)

            if retval and decoded_info:
                for i, info in enumerate(decoded_info):
                    if info:  # Only add successfully decoded QR codes
                        qr_codes.append((info, points[i]))

            # If no QR codes found with multi-detection, try single detection
            if not qr_codes:
                data, points, _ = self.qr_detector.detectAndDecode(image)
                if data:
                    qr_codes.append((data, points))

            # Always try different preprocessing techniques to find additional QR codes
            # Convert to grayscale first
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            found_texts = set([qr[0] for qr in qr_codes])  # Track what we've already found

            # Apply various preprocessing techniques
            techniques = []

            # Enhanced contrast using CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            clahe_enhanced = clahe.apply(gray)
            techniques.append(("CLAHE", clahe_enhanced))

            # Adaptive thresholding
            adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            techniques.append(("Adaptive", adaptive_thresh))

            # Otsu's thresholding (good for bimodal images)
            _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            techniques.append(("Otsu", otsu_thresh))

            # Try inverted Otsu for Instagram-style QR codes (white on black)
            _, otsu_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            techniques.append(("OtsuInv", otsu_inv))

            # Aggressive contrast enhancement for low contrast images
            contrast_enhanced = cv2.convertScaleAbs(gray, alpha=2.0, beta=50)
            techniques.append(("HighContrast", contrast_enhanced))

            # Gamma correction for very dark or very bright images
            gamma_dark = cv2.LUT(gray, np.array([((i / 255.0) ** (1.0 / 0.5)) * 255 for i in range(256)]).astype("uint8"))
            techniques.append(("GammaDark", gamma_dark))

            gamma_bright = cv2.LUT(gray, np.array([((i / 255.0) ** (1.0 / 1.5)) * 255 for i in range(256)]).astype("uint8"))
            techniques.append(("GammaBright", gamma_bright))

            # Bilateral filter for noise reduction while preserving edges
            bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
            techniques.append(("Bilateral", bilateral))

            # Create tasks for parallel processing
            tasks = []
            for scale in [1.0, 1.5]:  # Try at original scale and 1.5x scale
                for technique_name, processed_img in techniques:
                    tasks.append((technique_name, processed_img, scale, found_texts.copy()))

            # Process techniques in parallel using ThreadPoolExecutor
            max_workers = min(len(tasks), multiprocessing.cpu_count(), 8)  # Cap at 8 workers

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_task = {
                    executor.submit(self._apply_preprocessing_technique, task): task
                    for task in tasks
                }

                # Collect results
                for future in future_to_task:
                    try:
                        new_qr_codes = future.result()

                        # Add new unique QR codes
                        for qr_text, qr_points in new_qr_codes:
                            if qr_text not in found_texts:
                                found_texts.add(qr_text)
                                qr_codes.append((qr_text, qr_points))

                    except Exception as e:
                        continue  # Skip failed tasks

        except Exception as e:
            print(f"Error detecting QR codes: {e}")

        return qr_codes

    def process_single_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single file and return results.

        Args:
            file_path (str): Path to the image file

        Returns:
            Dict[str, Any]: Results containing filename and QR codes found
        """
        filename = os.path.basename(file_path)

        try:
            # Load image
            image = self.load_image(file_path)
            if image is None:
                return {
                    "filename": filename,
                    "success": False,
                    "error": "Failed to load image",
                    "qr_codes": []
                }

            # Detect QR codes
            qr_codes = self.detect_qr_codes(image)

            # Extract just the text data (not the points)
            qr_texts = [qr[0] for qr in qr_codes]

            return {
                "filename": filename,
                "success": True,
                "qr_codes": qr_texts,
                "count": len(qr_texts)
            }

        except Exception as e:
            return {
                "filename": filename,
                "success": False,
                "error": str(e),
                "qr_codes": []
            }


# FastAPI Application
app = FastAPI(
    title="QR Code Detector API",
    description="FastAPI service for detecting QR codes in uploaded images",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static directory if it doesn't exist
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global detector instance
detector = QRCodeDetector()


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
async def detect_qr_codes(files: List[UploadFile] = File(...)):
    """
    Detect QR codes in uploaded files.

    Args:
        files: List of uploaded files

    Returns:
        JSON response with detection results
    """
    import time
    start_time = time.time()

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    # Create temporary directory for uploaded files
    with tempfile.TemporaryDirectory() as temp_dir:
        file_paths = []

        # Save uploaded files to temporary directory
        for file in files:
            if not file.filename:
                continue

            # Validate file extension
            allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.heic', '.heif'}
            file_extension = Path(file.filename).suffix.lower()

            if file_extension not in allowed_extensions:
                continue  # Skip unsupported files

            file_path = os.path.join(temp_dir, file.filename)

            try:
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                file_paths.append(file_path)
            except Exception as e:
                print(f"Error saving file {file.filename}: {e}")
                continue

        if not file_paths:
            raise HTTPException(status_code=400, detail="No valid image files found")

        # Process files in parallel
        loop = asyncio.get_event_loop()
        max_workers = min(len(file_paths), multiprocessing.cpu_count(), 4)  # Limit concurrent processing

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all file processing tasks
            futures = [
                loop.run_in_executor(executor, detector.process_single_file, file_path)
                for file_path in file_paths
            ]

            # Wait for all results
            results = await asyncio.gather(*futures, return_exceptions=True)

        # Process results and handle exceptions
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                # Handle exceptions from parallel processing
                processed_results.append({
                    "filename": "unknown",
                    "success": False,
                    "error": str(result),
                    "qr_codes": []
                })
            else:
                processed_results.append(result)

    processing_time = time.time() - start_time

    return JSONResponse(content={
        "success": True,
        "processing_time": processing_time,
        "results": processed_results
    })


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "QR Code Detector API"}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="QR Code Detector FastAPI Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

    args = parser.parse_args()

    print("🚀 Starting QR Code Detector API...")
    print(f"📡 Server will be available at: http://{args.host}:{args.port}")
    print("🌐 Frontend: http://{args.host}:{args.port}/")
    print("📚 API Docs: http://{args.host}:{args.port}/docs")

    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )
