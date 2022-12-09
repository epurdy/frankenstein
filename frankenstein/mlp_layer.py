from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

def cluster_mlp_neurons(model, num_clusters):
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