import cv2
import numpy as np
import os

def create_dog_silhouette():
    # 1. Create a 512x512 White Canvas
    img = np.ones((512, 512, 3), dtype=np.uint8) * 255
    
    # Set drawing color to solid black
    color = (0, 0, 0)
    thickness = -1 # -1 means fill the shape

    # --- Draw a crude "Dog" shape using basic primitives ---
    
    # Body (Ellipse)
    cv2.ellipse(img, center=(256, 300), axes=(130, 80), angle=0, startAngle=0, endAngle=360, color=color, thickness=thickness)

    # Head (Circle)
    cv2.circle(img, center=(380, 240), radius=60, color=color, thickness=thickness)

    # Snout (Smaller Circle)
    cv2.circle(img, center=(435, 240), radius=35, color=color, thickness=thickness)

    # Ears (Triangles/Polygons)
    ear1 = np.array([[350, 190], [340, 120], [380, 180]], np.int32)
    ear2 = np.array([[390, 190], [420, 120], [410, 180]], np.int32)
    cv2.fillPoly(img, [ear1, ear2], color)

    # Legs (Thick lines)
    cv2.line(img, (200, 350), (190, 480), color, thickness=40) # Back Leg
    cv2.line(img, (340, 350), (350, 480), color, thickness=40) # Front Leg

    # Tail (Polyline)
    tail_pts = np.array([[130, 300], [90, 250], [80, 200]], np.int32)
    cv2.polylines(img, [tail_pts], isClosed=False, color=color, thickness=30)

    # 2. Save image
    filename = "test_silhouette_dog.png"
    cv2.imwrite(filename, img)
    print(f"Successfully created {filename}")
    return filename

if __name__ == "__main__":
    create_dog_silhouette()