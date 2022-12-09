from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import gaussian_kde

from tqdm import tqdm

from frankenstein.utils import get_device

def cluster_mlp_neurons(model, num_clusters):
    device = get_device(model)
    for layer in tqdm(range(model.cfg.n_layers)):
        sims = np.zeros((num_clusters, num_clusters))
        mlp_win = model.blocks[layer].mlp.W_in.T.detach().cpu().numpy()
        clustering = AgglomerativeClustering(n_clusters=num_clusters, affinity='cosine', 
            linkage='complete').fit(mlp_win)
        print(np.bincount(clustering.labels_))
        for i in range(num_clusters):
            for j in range(i, num_clusters):
                cosine_similarities = F.cosine_similarity(
                    torch.tensor(mlp_win[clustering.labels_ == i], device=device).unsqueeze(0),
                    torch.tensor(mlp_win[clustering.labels_ == j], device=device).unsqueeze(1),
                    dim=-1).detach().cpu().numpy().ravel()
                sims[i, j] = cosine_similarities.mean()
                sims[j, i] = sims[i, j]
        plt.figure(figsize=(10, 10))
        plt.title(f'Cosine similarity of mlp clusters for layer {layer}')
        plt.imshow(sims)
        plt.colorbar()

def measure_mlp_mask_values(model, texts):
    device = get_device(model)
    mask_values_by_layer = defaultdict(list)
    for text in tqdm(texts):
        tokens = model.tokenizer.encode(text)
        tokens = [50256] + tokens + [50256]
        tokens = tokens[:model.cfg.n_ctx]
        tokens = torch.tensor(tokens, device = device).unsqueeze(0)
        logits, cache = model.run_with_cache(tokens)
        for i in range(12):
            mask_values = cache[f'blocks.{i}.mlp.hook_post'] / cache[f'blocks.{i}.mlp.hook_pre']
            mask_values_by_layer[i].extend(mask_values.cpu().numpy().ravel())

    fig, subplots = plt.subplots(3, 4, figsize=(20, 20))
    for i, ax in enumerate(subplots.ravel()):
        subsample = np.random.choice(mask_values_by_layer[i], 100_000)
        print(i, len(mask_values_by_layer[i]))
        ax.set_title(f'Mask values of mlp.{i}')
        ax.hist(subsample, bins=300)


def eval_plot(all_evals, titles):
    fig, axes = plt.subplots(3, 4, subplot_kw=dict(polar=True), figsize=(100, 100))
    for (evals, ax, title) in zip(all_evals, axes.ravel(), titles):
        radius = np.abs(evals)
        theta = np.arctan2(evals.imag, evals.real)
        xy = np.vstack([evals.real, evals.imag])
        z = gaussian_kde(xy)(xy)
        ax.scatter(theta, radius, c=z, s=1000, cmap='viridis', linewidth=0)
        ax.set_rscale('symlog')
        ax.set_title(title)
        ax.set_rlim(0, 2 * radius.max())

def plot_mlp_eigenvalues(model):
    all_evals = []
    titles = []
    for layer in range(model.cfg.n_layers):  
        mlp = model.blocks[layer].mlp      
        win = mlp.W_in.detach().cpu().numpy()
        wout = mlp.W_out.detach().cpu().numpy()
        winout = win @ wout
        evals = np.linalg.eigvals(winout)
        all_evals.append(evals)
        titles.append(f'MLP evals, attn.{layer:02d}')
    eval_plot(all_evals, titles)