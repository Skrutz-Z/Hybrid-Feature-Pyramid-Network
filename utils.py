import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_sample_data():
    pyramid_input = [np.random.rand(64, 128, 128), np.random.rand(128, 64, 64), np.random.rand(256, 32, 32)]
    return pyramid_input

def visualize_prediction(hf_pyramids):
    data = hf_pyramids[0][0]  # Extract data for visualization
    plt.figure(figsize=(8, 6))
    plt.imshow(data, cmap='hot')
    plt.colorbar()
    plt.title("Detection Using HFPN")
    plt.figtext(0.5, 0.01, "Made by Shubh Rakesh Nahar Student ID - 1681276", ha="center", fontsize=8, color="gray")
    threshold = 0.8  # Define a threshold for object detection
    detected_coords = np.argwhere(data > threshold)

    for (x, y) in detected_coords:
        rect = patches.Rectangle((y - 1, x - 1), 2, 2, linewidth=1, edgecolor='blue', facecolor='none')
        plt.gca().add_patch(rect)

    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()

