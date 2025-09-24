import google.generativeai as genai
from PIL import Image
import os
from dotenv import load_dotenv
from pillow_heif import register_heif_opener

# Register the HEIF opener to enable support for .HEIC files
register_heif_opener()

# Load environment variables from a .env file
load_dotenv()

# Set your API key from the environment variables.
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError("API key not found. Please set the GEMINI_API_KEY in your .env file.")

genai.configure(api_key=api_key)

# Initialize the Gemini model with multimodal capabilities.
# gemini-1.5-pro is a good choice for this task.
model = genai.GenerativeModel('gemini-1.5-pro')

def get_qr_codes_from_image(image_path):
    """
    Analyzes an image for QR codes and returns their decoded links or data.

    Args:
        image_path (str): The file path to the image.

    Returns:
        str: A string containing the links/data from the QR codes, or an error message.
    """
    try:
        # Load the image using the Pillow library, now with .HEIC support
        img = Image.open(image_path)
        
        # Create a detailed prompt to instruct the model.
        prompt = """
        Analyze this image for any QR codes. For each QR code you find, 
        please decode the information it contains. List the decoded links 
        or text one per line. If you cannot find any QR codes, please state that.
        
        Example Output:
        https://example.com/page1
        https://example.com/page2
        
        Or:
        I could not find any QR codes in the image.
        """
        
        # Make the API call with both the prompt and the image.
        response = model.generate_content([prompt, img])
        
        return response.text
        
    except FileNotFoundError:
        return f"Error: The file at {image_path} was not found."
    except Exception as e:
        return f"An error occurred: {e}"

# --- Example Usage ---
# Use the correct file path.
image_file = 'Cornucopia 2025 Photos/IMG_2254.HEIC'

if __name__ == "__main__":
    print("--- QR Code Analysis Result ---")
    result = get_qr_codes_from_image(image_file)
    print(result)