import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class cos_simi_model():
    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))   
    
    def __init__(self, model_args, anchors, batch_size=256, is_binary=False):
        self.model = SentenceTransformer(model_args.model_name_or_path, device="cuda")
        self.model.eval()
        self.is_binary = is_binary

        # anchor is row text
        self.anchors = anchors
        self.batch_size = batch_size
        self.anchors_emb = self.model.encode(self.anchors, batch_size=self.batch_size, convert_to_numpy=True, show_progress_bar=True)
        

    def encode(self, sentences: list[str], **kwargs):
        self.model.eval()
        encoded = []
        for i in range(0, len(sentences), self.batch_size):
            
            batch = []
            if (len(sentences)-i) >= self.batch_size:
                batch = sentences[i:i+self.batch_size]
            else:
                batch = sentences[i:len(sentences)]

            row_embedding = self.model.encode(batch)

            cos_simi = cosine_similarity(row_embedding, self.anchors_emb)
            
            encoded.append(cos_simi)
            # encoded.append(row_embedding)
        stack_ed = np.vstack(encoded)

        if self.is_binary == True:
            if len(self.anchors) == 200:
                k = 20  
            elif len(self.anchors) == 500:
                k = 50
            binary_embeddings = np.zeros_like(stack_ed, dtype=int)
            for i in range(stack_ed.shape[0]):
                topk_indices = np.argsort(stack_ed[i])[-k:]
                binary_embeddings[i, topk_indices] = 1
            stack_ed = binary_embeddings

        return stack_ed
    
    
    def explain(self, embedding1, embedding2, num_explanations=None):
     
        product = embedding1 * embedding2
        count = np.sum(product)

        return count
    