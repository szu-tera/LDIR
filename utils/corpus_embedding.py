from sentence_transformers import SentenceTransformer
import numpy as np
import json

def main():
    corpus = []

    corpus_embeddings_dir = "/mnt/d/shenzhanyu/Interpre_emb/anchor-Interpre/data/meid2_all6M_embedding.npy"
    with open("/mnt/d/shenzhanyu/Interpre_emb/anchor-Interpre/data/medi2_documents-t.json", "r") as f:
        corpus = json.load(f)

    corpus_embeddings = []
    anchors_dict = {}
    
    embedding_model = SentenceTransformer("WhereIsAI/UAE-Large-V1")

    cuda_pool = embedding_model.start_multi_process_pool([0,1,2,3,4,5,6,7])

    corpus_embeddings = embedding_model.encode_multi_process(corpus, batch_size=1014, pool=cuda_pool, show_progress_bar=True)

    # corpus_embeddings = torch.tensor(corpus_embeddings).cuda()
    # corpus_embeddings = corpus_embeddings.unsqueeze(0)
    # print(corpus_embeddings.shape)


    # for i in selected_embedding_keys:
    #     anchors_dict[i] = corpus[i]
    np.save(corpus_embeddings_dir, corpus_embeddings)

if __name__ == '__main__':
    main()