# Example: How to use the latexp app in a Jupyter notebook
# 
# This file demonstrates how to use the build_explorer function in a notebook
# Copy and paste the relevant sections into your notebook cells

# ============================================================================
# CELL 1: Install required dependencies (run this first)
# ============================================================================
"""
!pip install jupyter-dash plotly numpy
"""

# ============================================================================
# CELL 2: Import the app and other required libraries
# ============================================================================
import numpy as np
from latexp.app import build_explorer

# ============================================================================
# CELL 3: Generate sample data (replace with your actual data)
# ============================================================================
# Generate sample data for demonstration
# In practice, you would load your actual images, latent vectors, etc.

# Sample parameters
n_samples = 1000
image_height, image_width = 28, 28  # Example: MNIST-like images

# Generate random sample data
np.random.seed(42)  # For reproducible results

# 1. Sample images (replace with your actual images)
images = np.random.rand(n_samples, image_height, image_width)

# 2. Sample latent vectors (2D for visualization)
latent_vectors = np.random.randn(n_samples, 2)

# 3. Sample cluster assignments (replace with your actual clustering results)
clusters = np.random.randint(0, 5, n_samples)

# 4. Sample label names mapping
label_names = {
    0: "Class A",
    1: "Class B", 
    2: "Class C",
    3: "Class D"
}

# 5. Sample assigned labels (replace with your actual labels)
# -1 means unlabeled
assigned_labels = np.random.choice([-1, 0, 1, 2, 3], n_samples, p=[0.3, 0.2, 0.2, 0.15, 0.15])

print(f"Generated {n_samples} samples")
print(f"Image shape: {images.shape}")
print(f"Latent vectors shape: {latent_vectors.shape}")
print(f"Unique clusters: {np.unique(clusters)}")
print(f"Unique labels: {np.unique(assigned_labels)}")

# ============================================================================
# CELL 4: Build and run the explorer app
# ============================================================================
# Build the explorer app
app = build_explorer(
    images=images,
    latent_vectors=latent_vectors,
    clusters=clusters,
    label_names=label_names,
    assigned_labels=assigned_labels
)

# Run the app in the notebook
# This will display the interactive dashboard
app.run_server(mode='inline', port=8050, debug=False)

# ============================================================================
# CELL 5: Alternative - Run in a separate tab (optional)
# ============================================================================
"""
# If you prefer to run the app in a separate browser tab:
app.run_server(mode='external', port=8050, debug=False)
"""

# ============================================================================
# CELL 6: Load real data example (replace the sample data above)
# ============================================================================
"""
# Example of how to load real data:

# Load images (example with numpy arrays)
# images = np.load('path/to/your/images.npy')  # Shape: (N, height, width)

# Load latent vectors (from your model)
# latent_vectors = np.load('path/to/your/latent_vectors.npy')  # Shape: (N, 2)

# Load cluster assignments (from clustering algorithm)
# clusters = np.load('path/to/your/clusters.npy')  # Shape: (N,)

# Define your label names
# label_names = {
#     0: "Your Class Name 1",
#     1: "Your Class Name 2",
#     # ... add more as needed
# }

# Load or initialize assigned labels
# assigned_labels = np.load('path/to/your/assigned_labels.npy')  # Shape: (N,)
# Or initialize with all unlabeled:
# assigned_labels = np.full(n_samples, -1)  # All unlabeled initially
"""

# ============================================================================
# CELL 7: Save updated labels (after using the app)
# ============================================================================
"""
# After using the app to assign labels, you can save the updated labels:
# np.save('path/to/save/updated_labels.npy', assigned_labels)

# Or save as CSV with indices:
# import pandas as pd
# df = pd.DataFrame({
#     'index': range(len(assigned_labels)),
#     'assigned_label': assigned_labels
# })
# df.to_csv('updated_labels.csv', index=False)
""" 