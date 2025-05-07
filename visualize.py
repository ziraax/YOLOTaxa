import os
import matplotlib.pyplot as plt
from collections import defaultdict
from config import CONFIG

def plot_underrepresented_classes(threshold=1000):
    """
    Plots only the classes from the processed dataset that have fewer than `threshold` images.
    """
    processed_root = CONFIG['processed_path']
    class_counts = defaultdict(int)

    if not os.path.exists(processed_root):
        print(f"[ERROR] Processed dataset path does not exist: {processed_root}")
        return

    for class_dir in os.listdir(processed_root):
        class_path = os.path.join(processed_root, class_dir)
        if os.path.isdir(class_path):
            count = len([
                f for f in os.listdir(class_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
            if count < threshold:
                class_counts[class_dir] = count

    if not class_counts:
        print(f"[INFO] No classes with fewer than {threshold} images found.")
        return

    # Sort classes by image count
    sorted_counts = sorted(class_counts.items(), key=lambda x: x[1])
    classes, counts = zip(*sorted_counts)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.bar(classes, counts, color='skyblue')
    plt.xticks(rotation=90)
    plt.title(f"Classes with Fewer than {threshold} Images")
    plt.ylabel("Number of Images")
    plt.xlabel("Class")
    plt.tight_layout()
    plt.show()


plot_underrepresented_classes(threshold=1500)