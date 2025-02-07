# %%

import os
import torch
from tqdm import tqdm

# Get all embedding files
embedding_dir = "/mnt/sdb/jengels/gpt2_embeddings/"
embedding_files = [f for f in os.listdir(embedding_dir) if f.startswith("embeddings_")]

# Read and concatenate all embeddings
embeddings_list = []
bar = tqdm(embedding_files)
for f in bar:
    path = os.path.join(embedding_dir, f)
    try:
        embeddings = torch.load(path).cpu()
        embeddings_list.append(embeddings)
        bar.set_description(f"Loaded {len(embeddings_list)}, each of shape {embeddings.shape}")
    except Exception as e:
        print(f"Error loading {f}: {e}")
        continue

# Concatenate into single tensor
all_embeddings = torch.cat(embeddings_list, dim=0)


# %%
num_samples = 10000000
random_indices = torch.randperm(all_embeddings.shape[0])[:num_samples]
random_embeddings = all_embeddings[random_indices]
# %%
import faiss
import numpy as np

# Convert to numpy and correct dtype for faiss
embeddings_np = random_embeddings.float().numpy()

# Number of clusters and dimensions
n_clusters = 768 * 4
d = embeddings_np.shape[1]

# Initialize and train k-means
kmeans = faiss.Kmeans(d, n_clusters, niter=20, verbose=True, gpu=True)
kmeans.train(embeddings_np)

# Get centroids and cluster assignments
centroids = kmeans.centroids
D, I = kmeans.index.search(embeddings_np, 1)  # Find nearest centroid for each point
cluster_assignments = I.ravel()

print(f"Performed k-means clustering with {n_clusters} clusters")
print(f"Centroids shape: {centroids.shape}")
print(f"Cluster assignments shape: {cluster_assignments.shape}")

# %%

# Convert centroids to tensor for easier computation
centroids_tensor = torch.from_numpy(centroids)

# Find closest centroid for each point
distances = torch.cdist(random_embeddings.float(), centroids_tensor.float())

closest_centroid_indices = torch.argmin(distances, dim=1)

# %%
# Normalize centroids
normalized_centroids = centroids_tensor / centroids_tensor.norm(dim=1, keepdim=True)

# Project embeddings onto closest normalized centroids
projected_embeddings = torch.zeros_like(random_embeddings)
for i in tqdm(range(len(random_embeddings))):
    centroid_idx = closest_centroid_indices[i]
    centroid = normalized_centroids[centroid_idx]
    # Project embedding onto its closest normalized centroid
    projection = (random_embeddings[i].float() @ centroid.float()) * centroid.float()
    projected_embeddings[i] = projection

# Compute FVU like in sae.py
total_variance = (random_embeddings - random_embeddings.mean(0)).pow(2).sum()
l2_loss = (projected_embeddings - random_embeddings).pow(2).sum()
fvu = l2_loss / total_variance

print(f"Fraction of variance unexplained: {fvu:.4f}")


# %%
