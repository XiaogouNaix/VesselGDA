import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np

def visualize_segmentation_with_points(segmentation_map, points, point_labels, point_gt_labels, segmentation_cmap='cool', alpha=0.6, title='Segmentation and Points Classification'):
    """Overlay graph nodes, predicted classes and reference classes on a segmentation image."""
    H, W = segmentation_map.shape
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    im = ax.imshow(segmentation_map, cmap=segmentation_cmap, alpha=alpha, extent=[0, W, H, 0])
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Segmentation Result', rotation=270, labelpad=15)
    points = np.array(points)
    point_labels = np.array(point_labels)
    points[:, 0] = np.clip(points[:, 0], 0, H - 1)
    points[:, 1] = np.clip(points[:, 1], 0, W - 1)
    point_colors = ['', 'blue', 'red', 'green']
    for class_idx in [1, 2, 3]:
        mask = point_labels + point_gt_labels * 2 == class_idx
        if np.any(mask):
            class_points = points[mask]
            ax.scatter(class_points[:, 1], class_points[:, 0], c=point_colors[class_idx], s=10, label=f'Class {class_idx}', edgecolors='white', linewidth=1)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    plt.tight_layout()
    return (fig, ax)
