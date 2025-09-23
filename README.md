# QR Code Detector for HEIC Images

A Python application that uses OpenCV to detect and decode QR codes in HEIC (High Efficiency Image Container) files and other common image formats.

## Features

- **HEIC Support**: Converts HEIC/HEIF files to OpenCV-compatible format
- **Multiple QR Code Detection**: Can detect and decode multiple QR codes in a single image
- **Visual Feedback**: Draws bounding boxes around detected QR codes with labels
- **Command Line Interface**: Easy-to-use CLI for batch processing
- **Flexible Output**: Save results to file and/or display on screen

## Installation

1. Clone or download this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

Or install packages individually:
```bash
pip install opencv-python numpy pillow pillow-heif
```

## Dependencies

- **opencv-python**: Computer vision library for QR code detection
- **numpy**: Numerical computing library
- **pillow**: Python Imaging Library for image processing
- **pillow-heif**: HEIC/HEIF image format support for Pillow

## Usage

### Command Line Interface

Basic usage:
```bash
python qr_code_detector.py path/to/your/image.heic
```

Save result to file:
```bash
python qr_code_detector.py path/to/your/image.heic -o result.jpg
```

Process without displaying the result:
```bash
python qr_code_detector.py path/to/your/image.heic --no-display
```

### Programmatic Usage

```python
from qr_code_detector import QRCodeDetector

# Create detector instance
detector = QRCodeDetector()

# Process an image
qr_data = detector.process_image(
    "path/to/image.heic",
    output_path="result.jpg",
    display=True
)

print(f"Detected QR codes: {qr_data}")
```

### Advanced Usage

```python
from qr_code_detector import QRCodeDetector

detector = QRCodeDetector()

# Load image manually
image = detector.load_image("image.heic")

# Detect QR codes
qr_codes = detector.detect_qr_codes(image)

# Process results
for i, (data, points) in enumerate(qr_codes):
    print(f"QR Code {i+1}: {data}")
    print(f"Corner points: {points}")

# Draw QR codes on image
result_image = detector.draw_qr_codes(image, qr_codes)
```

## Supported Image Formats

- **HEIC/HEIF**: High Efficiency Image Container format (iPhone photos)
- **JPEG/JPG**: Joint Photographic Experts Group
- **PNG**: Portable Network Graphics
- **BMP**: Bitmap
- **TIFF/TIF**: Tagged Image File Format
- **WEBP**: Web Picture format

## Examples

See `example_usage.py` for comprehensive usage examples including:
- Processing single HEIC images
- Batch processing multiple images
- Programmatic detection and analysis

## Troubleshooting

### HEIC Support Issues

If you encounter issues with HEIC files, ensure `pillow-heif` is properly installed:

```bash
pip uninstall pillow-heif
pip install pillow-heif
```

### Display Issues on macOS

If image display doesn't work properly on macOS, try:

```bash
pip uninstall opencv-python
pip install opencv-python-headless
```

### No QR Codes Detected

- Ensure QR codes are clearly visible and not too small
- Try adjusting image brightness/contrast
- Verify the QR code is not damaged or distorted

## Output

The detector will:
1. Print the number of QR codes found
2. Display the decoded data for each QR code
3. Show the image with QR codes highlighted (if display enabled)
4. Save the annotated image to file (if output path provided)

## Command Line Options

```
positional arguments:
  image_path            Path to the input image file

optional arguments:
  -h, --help            Show help message and exit
  -o OUTPUT, --output OUTPUT
                        Path to save the result image
  --no-display          Do not display the result image
```

## License

This project is open source and available under the MIT License.
