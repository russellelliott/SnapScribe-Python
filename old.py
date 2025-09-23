#!/usr/bin/env python3

import cv2
import numpy as np
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import argparse

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
    
    def detect_qr_codes(self, image: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """
        Detect and decode QR codes in an image.
        
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
            
            # If we found some QR codes but suspect there might be more, 
            # OR if we found no QR codes, try different preprocessing techniques
            if len(qr_codes) < 3:  # Assume there could be up to 3 QR codes in most images
                # Convert to grayscale first
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image
                
                found_texts = set([qr[0] for qr in qr_codes])  # Track what we've already found
                
                # Apply various preprocessing techniques (optimized list)
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
                
                # Try each technique at original scale and one additional scale
                for scale in [1.0, 1.5]:  # Only two scales to keep it fast
                    # Resize if needed
                    if scale != 1.0:
                        height, width = gray.shape[:2]
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                    
                    for technique_name, processed_img in techniques:
                        # Apply scale if needed
                        if scale != 1.0:
                            scaled_img = cv2.resize(processed_img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                        else:
                            scaled_img = processed_img
                            
                        try:
                            # Try multi-detection
                            retval, decoded_info, points, _ = self.qr_detector.detectAndDecodeMulti(scaled_img)
                            if retval and decoded_info:
                                for i, info in enumerate(decoded_info):
                                    if info and info not in found_texts:  # Avoid duplicates
                                        found_texts.add(info)
                                        # Scale points back if we resized the image
                                        if scale != 1.0:
                                            scaled_points = points[i] / scale
                                            qr_codes.append((info, scaled_points.astype(np.float32)))
                                        else:
                                            qr_codes.append((info, points[i]))
                            
                            # Also try single detection for this processed image
                            data, single_points, _ = self.qr_detector.detectAndDecode(scaled_img)
                            if data and data not in found_texts:
                                found_texts.add(data)
                                if scale != 1.0:
                                    scaled_points = single_points / scale
                                    qr_codes.append((data, scaled_points.astype(np.float32)))
                                else:
                                    qr_codes.append((data, single_points))
                                    
                        except Exception as e:
                            continue
        
        except Exception as e:
            print(f"Error detecting QR codes: {e}")
        
        return qr_codes
    
    def draw_qr_codes(self, image: np.ndarray, qr_codes: List[Tuple[str, np.ndarray]]) -> np.ndarray:
        """
        Draw bounding boxes around detected QR codes and add labels.
        
        Args:
            image (np.ndarray): OpenCV image array
            qr_codes (List[Tuple[str, np.ndarray]]): List of QR code data and points
            
        Returns:
            np.ndarray: Image with QR codes highlighted
        """
        result_image = image.copy()
        
        # Define colors for different QR codes (in BGR format)
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
        
        for i, (data, points) in enumerate(qr_codes):
            if points is not None and len(points) > 0:
                # Convert points to integer coordinates
                points = points.astype(int)
                
                # Use different colors for different QR codes
                color = colors[i % len(colors)]
                
                # Draw bounding box with thicker line
                cv2.polylines(result_image, [points], True, color, 4)
                
                # Add corner circles for better visibility
                for point in points:
                    cv2.circle(result_image, tuple(point), 8, color, -1)
                
                # Prepare label text
                if len(data) > 60:
                    label = f"QR {i+1}: {data[:57]}..."
                else:
                    label = f"QR {i+1}: {data}"
                
                # Find top-left corner for label placement
                top_left = tuple(points[0])
                
                # Calculate text size with larger font
                font_scale = 0.7
                font_thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                
                # Draw background rectangle for text with some padding
                padding = 5
                cv2.rectangle(result_image, 
                             (top_left[0] - padding, top_left[1] - text_height - padding - baseline),
                             (top_left[0] + text_width + padding, top_left[1] + padding),
                             color, -1)
                
                # Draw text with black color for contrast
                cv2.putText(result_image, label, 
                           (top_left[0], top_left[1] - baseline),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)
        
        # Add summary text at the bottom
        if qr_codes:
            height, width = result_image.shape[:2]
            summary_text = f"Total QR codes detected: {len(qr_codes)}"
            cv2.putText(result_image, summary_text, (10, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(result_image, summary_text, (10, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
        
        return result_image
    
    def process_image(self, image_path: str, output_path: Optional[str] = None, 
                     display: bool = True) -> List[str]:
        """
        Process an image file to detect and decode QR codes.
        
        Args:
            image_path (str): Path to the input image
            output_path (Optional[str]): Path to save the result image
            display (bool): Whether to display the result image
            
        Returns:
            List[str]: List of decoded QR code data
        """
        print(f"Processing image: {image_path}")
        
        # Load the image
        image = self.load_image(image_path)
        if image is None:
            return []
        
        # Detect QR codes
        qr_codes = self.detect_qr_codes(image)
        
        if not qr_codes:
            print("No QR codes found in the image.")
            return []
        
        print(f"Found {len(qr_codes)} QR code(s):")
        decoded_data = []
        
        for i, (data, _) in enumerate(qr_codes):
            print(f"  QR Code {i+1}: {data}")
            
            # Check if it's a URL and provide additional info
            if data.startswith(('http://', 'https://')):
                print(f"    Type: URL")
            elif data.startswith('mailto:'):
                print(f"    Type: Email")
            elif data.startswith('tel:'):
                print(f"    Type: Phone")
            elif data.startswith('wifi:'):
                print(f"    Type: WiFi")
            else:
                print(f"    Type: Text ({len(data)} characters)")
            
            decoded_data.append(data)
        
        # Draw QR codes on the image
        result_image = self.draw_qr_codes(image, qr_codes)
        
        # Save result image if output path provided
        if output_path:
            cv2.imwrite(output_path, result_image)
            print(f"Result saved to: {output_path}")
        
        # Display image if requested
        if display:
            self.display_image(result_image, f"QR Codes Detected: {len(qr_codes)}")
        
        return decoded_data
    
    def display_image(self, image: np.ndarray, window_title: str = "QR Code Detection"):
        """
        Display an image in a window.
        
        Args:
            image (np.ndarray): OpenCV image array
            window_title (str): Title for the display window
        """
        # Resize image if it's too large
        height, width = image.shape[:2]
        max_dimension = 1000
        
        if max(height, width) > max_dimension:
            if height > width:
                new_height = max_dimension
                new_width = int(width * (max_dimension / height))
            else:
                new_width = max_dimension
                new_height = int(height * (max_dimension / width))
            
            image = cv2.resize(image, (new_width, new_height))
        
        # Create a named window and make it resizable
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.imshow(window_title, image)
        
        print("Press any key in the image window to close, or press 'q' to quit...")
        
        # Wait for key press with better handling
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key != 255:  # Any key was pressed
                break
            # Check if window was closed
            if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
                break
        
        cv2.destroyAllWindows()


def main():
    """Main function to run the QR code detector from command line."""
    parser = argparse.ArgumentParser(description='Detect QR codes in HEIC and other image files')
    parser.add_argument('image_path', help='Path to the input image file')
    parser.add_argument('-o', '--output', help='Path to save the result image')
    parser.add_argument('--no-display', action='store_true', 
                       help='Do not display the result image')
    
    args = parser.parse_args()
    
    # Create detector instance
    detector = QRCodeDetector()
    
    # Process the image
    qr_data = detector.process_image(
        args.image_path,
        output_path=args.output,
        display=not args.no_display
    )
    
    if qr_data:
        print(f"\nSuccessfully detected {len(qr_data)} QR code(s)!")
    else:
        print("\nNo QR codes detected in the image.")


if __name__ == "__main__":
    main()