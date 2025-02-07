# %%

import os
import torch
from tqdm import tqdm
import einops
import faiss
import numpy as np
import matplotlib.pyplot as plt

# %%

# Get all embedding files
# embedding_dir = "/mnt/sdb/jengels/gpt2_embeddings/"
embedding_dir = "/mnt/sdb/jengels/pythia_embeddings/"
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
num_samples = 1000000
random_indices = torch.randperm(all_embeddings.shape[0])[:num_samples]
random_embeddings = all_embeddings[random_indices]
# %%

# Convert to numpy and correct dtype for faiss
embeddings_np = random_embeddings.float().numpy()

# Number of clusters and dimensions
n_clusters = 4096
d = embeddings_np.shape[1]

normalized_embeddings = embeddings_np / np.linalg.norm(embeddings_np, axis=1, keepdims=True)

# Initialize and train k-means
kmeans = faiss.Kmeans(d, n_clusters, niter=50, verbose=True, gpu=True, max_points_per_centroid=num_samples)
kmeans.train(normalized_embeddings)

# %%
# Get centroids and cluster assignments
centroids = torch.from_numpy(kmeans.centroids)
torch.save(centroids / centroids.norm(dim=1, keepdim=True), "centroids.pt")
torch.save(torch.from_numpy(normalized_embeddings), "normalized_embeddings.pt")
torch.save(torch.from_numpy(embeddings_np), "embeddings.pt")

# %%
device = "cuda:1"
centroids = torch.load("centroids.pt").to(device)
normalized_embeddings = torch.load("normalized_embeddings.pt").to(device)
embeddings = torch.load("embeddings.pt").to(device)

# Good: this doesn't work
# centroids = torch.rand_like(centroids)
# centroids = centroids / centroids.norm(dim=1, keepdim=True)

# Find closest centroid for each point
distances = einops.einsum(normalized_embeddings, centroids, "n d, c d -> n c")
closest_centroid_indices = torch.argmax(distances, dim=1)

closest_distances = distances.max(dim=1).values
print(closest_distances)
plt.hist(closest_distances.cpu().numpy(), bins=100)

# %%

# Gather the centroids for each embedding based on closest_centroid_indices
selected_centroids = centroids[closest_centroid_indices]  # Shape: [n, d]

# Compute dot products between embeddings and their closest centroids
dot_products = einops.einsum(embeddings, selected_centroids, "n d, n d -> n")
dot_products = dot_products.unsqueeze(-1)

# Scale the centroids by the dot products to get projections
projected_embeddings = dot_products * selected_centroids  # Shape: [n, d]

# Compute FVU like in sae.py
total_variance = (embeddings - embeddings.mean(0)).pow(2).sum()
l2_loss = (projected_embeddings - embeddings).pow(2).sum()
fvu = l2_loss / total_variance

print(f"Fraction of variance unexplained: {fvu:.4f}")

# %%
variance_explained = (
    1
    - ((embeddings - projected_embeddings) ** 2).sum()
    / (embeddings**2).sum()
)
print(f"Fraction of variance unexplained: {variance_explained.mean():.4f}")

# %%

# Compute FVU using residual sum of squares method
resid_sum_of_squares = (
    (embeddings - projected_embeddings).pow(2).sum(dim=-1)
)
total_sum_of_squares = (
    (embeddings - embeddings.mean(dim=0)).pow(2).sum(-1)
)

explained_variance = 1 - resid_sum_of_squares / total_sum_of_squares
fvu_alt = resid_sum_of_squares / total_sum_of_squares

print(f"Alternative FVU calculation: {fvu_alt.mean():.4f}")
print(f"Explained variance: {explained_variance.mean():.4f}")



# %%
