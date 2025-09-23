#!/usr/bin/env python3
"""
Test script to verify QR code detector functionality
"""

import sys
import os

def test_imports():
    """Test that all required packages can be imported."""
    try:
        import cv2
        print("✓ OpenCV imported successfully")
        print(f"  OpenCV version: {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
        print(f"  NumPy version: {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print("✓ PIL/Pillow imported successfully")
    except ImportError as e:
        print(f"✗ PIL/Pillow import failed: {e}")
        return False
    
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
        print("✓ pillow-heif imported successfully")
        print("  HEIC support enabled")
    except ImportError as e:
        print(f"✗ pillow-heif import failed: {e}")
        print("  HEIC support disabled")
    
    return True

def test_qr_detector():
    """Test the QR code detector class."""
    try:
        from qr_code_detector import QRCodeDetector
        detector = QRCodeDetector()
        print("✓ QRCodeDetector class imported and instantiated successfully")
        return True
    except Exception as e:
        print(f"✗ QRCodeDetector test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing QR Code Detector Setup")
    print("=" * 40)
    
    # Test imports
    print("Testing package imports:")
    imports_ok = test_imports()
    print()
    
    # Test detector class
    print("Testing QRCodeDetector class:")
    detector_ok = test_qr_detector()
    print()
    
    # Overall result
    if imports_ok and detector_ok:
        print("✓ All tests passed! The QR code detector is ready to use.")
        print("\nTo test with an actual image, use:")
        print("python qr_code_detector.py path/to/your/image.heic")
    else:
        print("✗ Some tests failed. Please check the installation.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
