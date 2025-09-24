import google.generativeai as genai
from PIL import Image
import os
import concurrent.futures
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
        tuple: A tuple containing the original file path and a string with the results.
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
        
        return image_path, response.text
        
    except FileNotFoundError:
        return image_path, "Error: File not found."
    except Exception as e:
        return image_path, f"An error occurred: {e}"

def process_folder_for_qr_codes(folder_path):
    """
    Goes through all image files in a folder and gets their QR code links
    using parallel processing.

    Args:
        folder_path (str): The path to the folder.
    """
    # Define a list of common image extensions to check for.
    image_extensions = ('.heic', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
    
    # Check if the folder exists.
    if not os.path.isdir(folder_path):
        print(f"Error: The folder at '{folder_path}' does not exist.")
        return

    print(f"--- Processing all image files in '{folder_path}' using parallelization ---")
    
    # Get all the file names in the specified folder and sort them for consistent order.
    all_files = os.listdir(folder_path)
    all_files.sort()
    
    # Filter for valid image files and create a list of full paths.
    image_paths = [
        os.path.join(folder_path, filename)
        for filename in all_files
        if os.path.isfile(os.path.join(folder_path, filename)) and filename.lower().endswith(image_extensions)
    ]
    
    if not image_paths:
        print("No image files found in the folder.")
        return

    # Use a ThreadPoolExecutor to run tasks in parallel.
    # The 'with' statement ensures the threads are properly managed and cleaned up.
    # We'll use a maximum of 5 workers for this example to avoid overwhelming the API or your machine.
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Create a mapping of Future objects to their original file paths.
        future_to_path = {
            executor.submit(get_qr_codes_from_image, path): path
            for path in image_paths
        }
        
        # Iterate over the completed tasks as they finish.
        for future in concurrent.futures.as_completed(future_to_path):
            file_path = future_to_path[future]
            try:
                # The result is a tuple: (original_path, result_string)
                _, result_text = future.result()
                print(f"\n--- Analysis for: {os.path.basename(file_path)} ---")
                print(result_text)
            except Exception as exc:
                print(f"{os.path.basename(file_path)} generated an exception: {exc}")

# --- Example Usage ---
# Set the target folder path.
target_folder = 'Cornucopia 2025 Photos'

if __name__ == "__main__":
    process_folder_for_qr_codes(target_folder)