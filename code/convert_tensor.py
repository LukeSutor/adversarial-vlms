import os
import sys
from utils import save_tensor_as_png

def main():
    # Specify the tensor file path
    tensor_path = "/blue/rcstudents/luke.sutor/adversarial-vlms/images/attack/llama-3.2-11b-vision-instruct/boiling.pt"
    
    # Check if the file exists
    if not os.path.exists(tensor_path):
        print(f"Error: File not found at {tensor_path}")
        return
    
    try:
        # Convert the tensor to PNG
        png_path = save_tensor_as_png(tensor_path)
        print(f"Successfully converted tensor to PNG: {png_path}")
    except Exception as e:
        print(f"Error converting tensor to PNG: {e}")

if __name__ == "__main__":
    main()
