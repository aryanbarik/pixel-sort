import cv2
import numpy as np
import os

def edge_guided_pixel_sort(img, canny_low=50, canny_high=150, sort_key='brightness'):
    """Perform edge-guided pixel sorting on an image."""
    # Convert image to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Edge detection (Canny)
    edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), canny_low, canny_high)
    
    # Choose sorting criterion
    if sort_key == 'hue':
        sort_channel = h
    elif sort_key == 'saturation':
        sort_channel = s
    else:  # brightness
        sort_channel = v

    # Process each row
    for y in range(img.shape[0]):
        row_edges = np.where(edges[y, :] > 0)[0]
        segments = np.split(np.arange(img.shape[1]), row_edges)
        
        for seg in segments:
            if len(seg) < 2: continue  # Skip single-pixel segments
        
            # Get slice indices
            start, end = seg[0], seg[-1]+1
            
            # Sort pixels in this segment
            sorted_order = np.argsort(sort_channel[y, start:end])
            v[y, start:end] = v[y, start:end][sorted_order]
            h[y, start:end] = h[y, start:end][sorted_order]
            s[y, start:end] = s[y, start:end][sorted_order]

    # Reconstruct and return the sorted image
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

def process_folder(input_folder, output_folder, canny_low=50, canny_high=150, sort_key='brightness'):
    """Process all images in a folder and save sorted images to another folder."""
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get list of image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    # Process each image
    for image_file in image_files:
        # Load image
        input_path = os.path.join(input_folder, image_file)
        img = cv2.imread(input_path)
        if img is None:
            print(f"Warning: Could not load image {image_file}. Skipping.")
            continue
        
        # Apply pixel sorting
        sorted_img = edge_guided_pixel_sort(img, canny_low, canny_high, sort_key)
        
        # Save the sorted image
        output_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_path, sorted_img)
        print(f"Processed and saved: {output_path}")

if __name__ == "__main__":
    # Example usage
    input_folder = "input"  # Folder containing input images
    output_folder = "sorted_images"  # Folder to save sorted images
    process_folder(input_folder, output_folder, canny_low=30, canny_high=100, sort_key='hue')