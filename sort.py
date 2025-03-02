import cv2
import numpy as np

def edge_guided_pixel_sort(image_path, output_path, 
                         canny_low=50, canny_high=150,
                         sort_key='brightness'):
    # Load image and convert to HSV
    img = cv2.imread(image_path)
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

    # Reconstruct and save
    result = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
    cv2.imwrite(output_path, result)

# Usage example
edge_guided_pixel_sort('371470.jpg', 'sorted_output3.jpg', 
                     canny_low=100, canny_high=150,
                     sort_key='hue')