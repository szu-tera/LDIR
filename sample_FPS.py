import torch
import numpy as np
from sentence_transformers import SentenceTransformer

import os
import json
import logging

logger = logging.getLogger(__name__)

def distance_fun(xyz, centroid):
    return torch.sum((xyz - centroid) ** 2, -1)

def farthest_point_sample_with_cuda(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    
    distance = torch.ones(B, N).to(device) * 1e10
    
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    selected_embedding = {}

    for i in range(npoint):
    
        centroids[:, i] = farthest
        
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 1024)
        
        dist = distance_fun(xyz, centroid)
        
        mask = dist < distance
        distance[mask] = dist[mask]
        
        farthest = torch.max(distance, -1)[1]

        selected_embedding[farthest.cpu().item()] = xyz[batch_indices, farthest, :].view(B, 1, 1024).squeeze().cpu().numpy()
    
    return centroids, selected_embedding


def main():

    devices_list = []
    devices_list.append(f"cuda:{0}")

    anchors_dict = {}
    anchor_file = f"./data/anchors/your_anchors.json"
    corpus_file = f"../data/medi2_documents-t.json"
    corpus_emb_file = "../data/medi2_documents_dense_embedding.npy"
    if not os.path.exists(anchor_file):

        
        corpus_embeddings = []
        corpus = [] 
        if not os.path.exists(corpus_emb_file):
            logger.info("corpus embedding")
            
            with open(corpus_file, "r") as f:
                corpus = json.load(f)

            embedding_model = SentenceTransformer("WhereIsAI/UAE-Large-V1")

            corpus_embeddings = embedding_model(corpus, batch_size=1024, show_progress_bar=True)
            
            np.save(corpus_emb_file, corpus_embeddings)
        else:
            with open(corpus_file, "r") as f:
                corpus = json.load(f)
                # corpus is dict
                corpus = list(corpus.values())
                corpus_embeddings = np.load(corpus_emb_file)
        
        
        logger.info("anchor processing")

        corpus_embeddings = torch.tensor(corpus_embeddings).cuda()
        corpus_embeddings = corpus_embeddings.unsqueeze(0)
        print("corpus_embeddings: ", corpus_embeddings.shape)

        index, selected_embedding = farthest_point_sample_with_cuda(corpus_embeddings, 500)

        selected_embedding_keys = list(selected_embedding.keys())

        for i in selected_embedding_keys:
            anchors_dict[i] = corpus[i]
        
        with open(anchor_file, "w") as f:
            json.dump(anchors_dict, f)

        logger.info("anchors have selected")
    
if __name__ == "__main__":
    main()

