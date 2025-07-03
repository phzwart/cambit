#!/usr/bin/env python3
"""
Debug app to test heatmap functionality on port 8055
"""

import numpy as np
from latexp import build_explorer

if __name__ == "__main__":
    # Generate test data
    n_samples = 100
    image_height, image_width = 28, 28
    
    print("Generating test data...")
    
    # Sample images (random data) - RGB format
    images = np.random.rand(n_samples, image_height, image_width, 3)
    
    # Sample latent vectors (2D for visualization)
    latent_vectors = np.random.randn(n_samples, 2)
    
    # Sample cluster assignments
    clusters = np.random.randint(0, 5, n_samples)
    
    # Sample label names mapping
    label_names = {
        0: "Class A",
        1: "Class B", 
        2: "Class C",
        3: "Class D",
        4: "Class E"
    }
    
    # Sample assigned labels (all unlabeled to start)
    assigned_labels = np.full(n_samples, -1)
    
    print(f"Generated {n_samples} samples")
    print(f"Image shape: {images.shape}")
    print(f"Latent vectors shape: {latent_vectors.shape}")
    print(f"Unique clusters: {np.unique(clusters)}")
    
    # Build the explorer app
    print("Building app...")
    app = build_explorer(
        images=images,
        latent_vectors=latent_vectors,
        clusters=clusters,
        label_names=label_names,
        assigned_labels=assigned_labels
    )
    
    # Run the app on port 8055
    print("Starting app on port 8055...")
    print("Open your browser to: http://127.0.0.1:8055")
    app.run_app(mode='external', port=8055, debug=True) 