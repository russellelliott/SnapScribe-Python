#!/usr/bin/env python3

import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Tuple, Set
import multiprocessing

def apply_preprocessing_technique(args) -> Tuple[str, np.ndarray, List[Tuple[str, np.ndarray]]]:
    """
    Apply a single preprocessing technique and detect QR codes.
    Returns (technique_name, processed_image, detected_qr_codes)
    """
    technique_name, processed_img, scale, qr_detector, found_texts = args
    
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
        retval, decoded_info, points, _ = qr_detector.detectAndDecodeMulti(scaled_img)
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
        data, single_points, _ = qr_detector.detectAndDecode(scaled_img)
        if data and data not in found_texts:
            if scale != 1.0:
                scaled_points = single_points / scale
                new_qr_codes.append((data, scaled_points.astype(np.float32)))
            else:
                new_qr_codes.append((data, single_points))
                
    except Exception as e:
        pass  # Continue on error
    
    return technique_name, processed_img, new_qr_codes

def detect_qr_codes_parallel(image: np.ndarray, qr_detector) -> List[Tuple[str, np.ndarray]]:
    """
    Parallel version of QR code detection with preprocessing techniques.
    """
    qr_codes = []
    
    try:
        # First, try to detect multiple QR codes at once (original method)
        retval, decoded_info, points, _ = qr_detector.detectAndDecodeMulti(image)
        
        if retval and decoded_info:
            for i, info in enumerate(decoded_info):
                if info:  # Only add successfully decoded QR codes
                    qr_codes.append((info, points[i]))
        
        # If no QR codes found with multi-detection, try single detection
        if not qr_codes:
            data, points, _ = qr_detector.detectAndDecode(image)
            if data:
                qr_codes.append((data, points))
        
        # Prepare for parallel preprocessing
        if len(qr_codes) < 3:  # Or remove this condition entirely
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            found_texts = set([qr[0] for qr in qr_codes])
            
            # Prepare all preprocessing techniques
            techniques = []
            
            # Enhanced contrast using CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            clahe_enhanced = clahe.apply(gray)
            techniques.append(("CLAHE", clahe_enhanced))
            
            # Adaptive thresholding 
            adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            techniques.append(("Adaptive", adaptive_thresh))
            
            # Otsu's thresholding
            _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            techniques.append(("Otsu", otsu_thresh))
            
            # Inverted Otsu
            _, otsu_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            techniques.append(("OtsuInv", otsu_inv))
            
            # Contrast enhancement
            contrast_enhanced = cv2.convertScaleAbs(gray, alpha=2.0, beta=50)
            techniques.append(("HighContrast", contrast_enhanced))
            
            # Gamma corrections
            gamma_dark = cv2.LUT(gray, np.array([((i / 255.0) ** (1.0 / 0.5)) * 255 for i in range(256)]).astype("uint8"))
            techniques.append(("GammaDark", gamma_dark))
            
            gamma_bright = cv2.LUT(gray, np.array([((i / 255.0) ** (1.0 / 1.5)) * 255 for i in range(256)]).astype("uint8"))
            techniques.append(("GammaBright", gamma_bright))
            
            # Bilateral filter
            bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
            techniques.append(("Bilateral", bilateral))
            
            # Create tasks for parallel processing
            tasks = []
            for scale in [1.0, 1.5]:
                for technique_name, processed_img in techniques:
                    tasks.append((technique_name, processed_img, scale, qr_detector, found_texts))
            
            # Process all techniques in parallel
            max_workers = min(len(tasks), multiprocessing.cpu_count())
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_task = {
                    executor.submit(apply_preprocessing_technique, task): task 
                    for task in tasks
                }
                
                # Collect results
                for future in future_to_task:
                    try:
                        technique_name, processed_img, new_qr_codes = future.result()
                        
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

# Example usage showing the performance difference
if __name__ == "__main__":
    import time
    
    # This would be integrated into your QRCodeDetector class
    detector = cv2.QRCodeDetector()
    
    # Load a test image (you'd replace this with your actual image loading)
    # image = cv2.imread("test_image.jpg")
    
    print("Parallel preprocessing can significantly speed up QR detection")
    print("when multiple preprocessing techniques need to be applied.")
    print("\nBenefits:")
    print("- CPU cores can work on different techniques simultaneously")
    print("- Reduces total processing time from sequential to parallel")
    print("- Especially beneficial for complex images requiring many techniques")
