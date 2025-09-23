#!/usr/bin/env python3
"""
Example usage of the QR Code Detector

This script demonstrates how to use the QRCodeDetector class
to detect QR codes in HEIC and other image files.
"""

from qr_code_detector import QRCodeDetector
import os
import sys


def main():
    """Example usage of the QR code detector."""
    
    # Create detector instance
    detector = QRCodeDetector()
    
    # Example 1: Process a single HEIC image
    print("=== Example 1: Processing a HEIC image ===")
    heic_path = "sample_image.heic"  # Replace with actual path
    
    if os.path.exists(heic_path):
        qr_data = detector.process_image(
            heic_path,
            output_path="result_with_qr_codes.jpg",
            display=True  # Set to False if running headless
        )
        
        print(f"Detected QR codes: {qr_data}")
    else:
        print(f"Sample HEIC file not found: {heic_path}")
        print("Please provide a valid HEIC file path")
    
    # Example 2: Process multiple images
    print("\n=== Example 2: Processing multiple images ===")
    image_extensions = ['.jpg', '.jpeg', '.png', '.heic', '.heif', '.bmp', '.tiff']
    current_dir = os.getcwd()
    
    image_files = []
    for ext in image_extensions:
        for file in os.listdir(current_dir):
            if file.lower().endswith(ext.lower()):
                image_files.append(file)
    
    if image_files:
        print(f"Found {len(image_files)} image file(s) in current directory:")
        for i, image_file in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {image_file}")
            qr_data = detector.process_image(
                image_file,
                output_path=f"result_{i+1}_{image_file}",
                display=False  # Don't display when processing multiple files
            )
            if qr_data:
                print(f"  Found QR codes: {qr_data}")
            else:
                print(f"  No QR codes found")
            print()
    else:
        print("No image files found in current directory")
    
    # Example 3: Load and analyze without display
    print("\n=== Example 3: Programmatic usage ===")
    sample_image = "test_qr.jpg"  # Replace with actual path
    
    if os.path.exists(sample_image):
        # Load image
        image = detector.load_image(sample_image)
        if image is not None:
            # Detect QR codes
            qr_codes = detector.detect_qr_codes(image)
            
            # Process results
            if qr_codes:
                print(f"Found {len(qr_codes)} QR code(s):")
                for i, (data, points) in enumerate(qr_codes):
                    print(f"  QR Code {i+1}:")
                    print(f"    Data: {data}")
                    print(f"    Corner points: {points.shape if points is not None else 'None'}")
            else:
                print("No QR codes detected")
    else:
        print(f"Test image not found: {sample_image}")


if __name__ == "__main__":
    main()
