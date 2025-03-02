import cv2
import numpy as np
import os

def edge_guided_pixel_sort(img, canny_low=50, canny_high=150, sort_key='brightness', sort_method='row'):
    """Perform edge-guided pixel sorting on an image with optional sorting methods."""
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

    # Apply the selected sorting method
    if sort_method == 'row':
        # Row-based sorting (default)
        v, h, s = _sort_rows(v, h, s, edges, sort_channel)
    elif sort_method == 'column':
        # Column-based sorting (multi-directional)
        v, h, s = _sort_columns(v, h, s, edges, sort_channel)
    elif sort_method == 'gradient':
        # Gradient-based sorting
        v, h, s = _sort_gradient(img, v, h, s, edges, sort_channel)
    elif sort_method == 'region':
        # Region-based sorting
        v, h, s = _sort_regions(v, h, s, edges, sort_channel)
    else:
        raise ValueError(f"Invalid sort_method: {sort_method}. Choose from 'row', 'column', 'gradient', or 'region'.")

    # Reconstruct and return the sorted image
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

def _sort_rows(v, h, s, edges, sort_channel):
    """Sort pixels row-wise within edge-bounded segments."""
    for y in range(v.shape[0]):
        row_edges = np.where(edges[y, :] > 0)[0]
        segments = np.split(np.arange(v.shape[1]), row_edges)
        
        for seg in segments:
            if len(seg) < 2: continue  # Skip single-pixel segments
        
            # Get slice indices
            start, end = seg[0], seg[-1]+1
            
            # Sort pixels in this segment
            sorted_order = np.argsort(sort_channel[y, start:end])
            v[y, start:end] = v[y, start:end][sorted_order]
            h[y, start:end] = h[y, start:end][sorted_order]
            s[y, start:end] = s[y, start:end][sorted_order]
    return v, h, s

def _sort_columns(v, h, s, edges, sort_channel):
    """Sort pixels column-wise within edge-bounded segments."""
    # Transpose the image for column-wise processing
    v_t = v.T
    h_t = h.T
    s_t = s.T
    edges_t = edges.T
    
    for x in range(v_t.shape[0]):
        col_edges = np.where(edges_t[x, :] > 0)[0]
        segments = np.split(np.arange(v_t.shape[1]), col_edges)
        
        for seg in segments:
            if len(seg) < 2: continue  # Skip single-pixel segments
        
            # Get slice indices
            start, end = seg[0], seg[-1]+1
            
            # Sort pixels in this segment
            sorted_order = np.argsort(sort_channel[start:end, x])
            v_t[x, start:end] = v_t[x, start:end][sorted_order]
            h_t[x, start:end] = h_t[x, start:end][sorted_order]
            s_t[x, start:end] = s_t[x, start:end][sorted_order]
    
    # Transpose back to original orientation
    return v_t.T, h_t.T, s_t.T

def _sort_gradient(img, v, h, s, edges, sort_channel):
    """Sort pixels based on gradient direction."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_dir = np.arctan2(sobely, sobelx)
    
    # Sort pixels based on gradient direction
    for y in range(v.shape[0]):
        for x in range(v.shape[1]):
            if edges[y, x] > 0: continue  # Skip edges
            
            # Determine sorting direction based on gradient
            if gradient_dir[y, x] > 0:
                # Sort horizontally
                row_edges = np.where(edges[y, :] > 0)[0]
                segments = np.split(np.arange(v.shape[1]), row_edges)
                for seg in segments:
                    if len(seg) < 2: continue
                    start, end = seg[0], seg[-1]+1
                    sorted_order = np.argsort(sort_channel[y, start:end])
                    v[y, start:end] = v[y, start:end][sorted_order]
                    h[y, start:end] = h[y, start:end][sorted_order]
                    s[y, start:end] = s[y, start:end][sorted_order]
            else:
                # Sort vertically
                col_edges = np.where(edges[:, x] > 0)[0]
                segments = np.split(np.arange(v.shape[0]), col_edges)
                for seg in segments:
                    if len(seg) < 2: continue
                    start, end = seg[0], seg[-1]+1
                    sorted_order = np.argsort(sort_channel[start:end, x])
                    v[start:end, x] = v[start:end, x][sorted_order]
                    h[start:end, x] = h[start:end, x][sorted_order]
                    s[start:end, x] = s[start:end, x][sorted_order]
    return v, h, s

def _sort_regions(v, h, s, edges, sort_channel):
    """Sort pixels within edge-bounded regions while preserving local structure."""
    # Preprocess edges to ensure closed regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Find regions using 8-connected components on inverted edges
    _, labels = cv2.connectedComponents(255 - dilated_edges, connectivity=8)
    
    # Process each region
    for label in np.unique(labels):
        if label == 0:  # Background
            continue
            
        # Create region mask and get bounding box
        mask = (labels == label).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Skip small regions
        if cv2.contourArea(contours[0]) < 100:
            continue

        # Get region boundaries
        x,y,w,h_region = cv2.boundingRect(contours[0])
        
        # Process each row in the region
        for row in range(y, y+h_region):
            # Get row slice within region bounds
            row_mask = mask[row, x:x+w]
            if np.sum(row_mask) == 0:
                continue
                
            # Find start/end of region segment in this row
            cols = np.where(row_mask > 0)[0]
            if len(cols) == 0:
                continue
                
            start_col = cols[0] + x
            end_col = cols[-1] + x + 1
            
            # Split into sub-segments using original edges
            edge_positions = np.where(edges[row, start_col:end_col] > 0)[0] + start_col
            segments = np.split(np.arange(start_col, end_col), edge_positions - start_col)
            
            # Sort each sub-segment
            for seg in segments:
                if len(seg) < 2:
                    continue
                    
                # Get absolute column indices
                start = seg[0]
                end = seg[-1] + 1
                
                # Sort within the segment
                sorted_order = np.argsort(sort_channel[row, start:end])
                v[row, start:end] = v[row, start:end][sorted_order]
                h[row, start:end] = h[row, start:end][sorted_order]
                s[row, start:end] = s[row, start:end][sorted_order]

    return v, h, s

def process_folder(input_folder, output_folder, canny_low=50, canny_high=150, sort_key='brightness', sort_method='row'):
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
        sorted_img = edge_guided_pixel_sort(img, canny_low, canny_high, sort_key, sort_method)
        
        # Save the sorted image
        output_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_path, sorted_img)
        print(f"Processed and saved: {output_path}")

if __name__ == "__main__":
    # Example usage
    input_folder = "input"  # Folder containing input images
    output_folder = "sorted_images"  # Folder to save sorted images
    process_folder(input_folder, output_folder, canny_low=100, canny_high=200, sort_key='hue', sort_method='region')