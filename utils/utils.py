import os
import json
from tqdm import tqdm

from beir import util
from sklearn.preprocessing import normalize
from sklearn.metrics import ndcg_score
from beir.datasets.data_loader import GenericDataLoader
import numpy as np
import logging
import random

def gather_result_sts(result_path):
    result_dir = os.path.join(result_path, "no_model_name_available/no_revision_available")
    files = os.listdir(result_dir)

    irre = ("STS17","STS22","STSBenchmarkMultilingualSTS")
    result_files = []

    for file in files:
        if file.endswith(".json") and file.startswith(("STS","SICK")) and not file.startswith(irre):
            result_files.append(file)

    result_dic = {}
    for file in result_files:

        with open(os.path.join(result_dir, file), "r") as f:
            
            result = json.load(f)
            result_dic[result["task_name"]] = result["scores"]["test"][0]["spearman"] * 100

    result_dic_keys = list(result_dic.keys())
    sum = 0.0

    for i in result_dic_keys:
        sum += result_dic[i]
    aver_result = sum/len(result_dic_keys)
    result_dic["Avg."] = aver_result

    with open(os.path.join(result_dir, "a_sts_sum.json"), "w") as f:
        json.dump(result_dic, f, indent=2, sort_keys=True)

def gather_result_retrie_or_cluste(result_path, task_type):
    
    
    if task_type == "Clustering":
        measure = "v_measure"
    elif task_type == "Retrieval":
        measure = "ndcg_at_10"
    result_dir = os.path.join(result_path, "no_model_name_available/no_revision_available")
    files = os.listdir(result_dir)

    result_files = []
    for file in files:
        if file.endswith(".json") and not file.startswith("model"):
            result_files.append(file)
    result_dic = {}
    for file in result_files:
        if "msmarco" in file:
            with open(os.path.join(result_dir, file), "r") as f:
                result = json.load(f)
                result_dic["msmarco"] = result["msmarco"]*100
        else:
            with open(os.path.join(result_dir, file), "r") as f:
                result = json.load(f)
                result_dic[result["task_name"]] = result["scores"]["test"][0][measure] * 100

    result_dic_keys = list(result_dic.keys())
    # print("****************", result_dic_keys)
    sum = 0.0
    for i in result_dic_keys:
        sum += result_dic[i]
    aver_result = sum/len(result_dic_keys)
    result_dic["Avg."] = aver_result

    with open(os.path.join(result_dir, "a_sts_sum.json"), "w") as f:
        json.dump(result_dic, f, indent=2, sort_keys=True)


class evaluate_msmarco():
    
    def __init__(self,model_args):
        self.SAMPLING_RATIO = 0.01
        self.batch_size = model_args.batch_size
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        dataset = "msmarco"
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
        out_dir = os.path.join("/mnt/d/shenzhanyu/Interpre_emb/anchor-Interpre", "data/BEIR")
        data_path = util.download_and_unzip(url, out_dir)

        corpus, self.queries, self.qrels = GenericDataLoader(data_folder=data_path).load(split="dev")

        self.logger.info(f"Loaded {len(corpus)} documents")

        gt_docs = set()
        for k in self.qrels:
            for d in self.qrels[k]:
                gt_docs.add(d)
        len(gt_docs)

        # subsample 1% of the corpus, but keep all the gt_docs
        # Set a seed for reproducibility
        random.seed(42)

        selected_corpus = set(gt_docs)  # Start with all ground truth documents

        # Calculate how many additional documents we need to reach 1% of the corpus
        target_size = int(len(corpus) * self.SAMPLING_RATIO) # Change this to test a subset only
        additional_docs_needed = max(0, target_size - len(selected_corpus))

        # Randomly select additional documents from the corpus
        remaining_docs = list(set(corpus.keys()) - selected_corpus)  # Convert to list
        additional_docs = random.sample(remaining_docs, min(additional_docs_needed, len(remaining_docs)))

        selected_corpus.update(additional_docs)

        # Create a new corpus dictionary with only the selected documents
        self.subsampled_corpus = {doc_id: corpus[doc_id] for doc_id in selected_corpus}

        self.logger.info(f"Original corpus size: {len(corpus)}")
        self.logger.info(f"Subsampled corpus size: {len(self.subsampled_corpus)}")
        self.logger.info(f"Percentage of original corpus: {len(self.subsampled_corpus) / len(corpus) * 100:.2f}%")
        self.logger.info(f"Number of ground truth documents: {len(gt_docs)}")
        self.logger.info(f"All ground truth documents included: {set(gt_docs).issubset(set(self.subsampled_corpus.keys()))}")
    
    def run_evaluation(self, model):
        # Extract query IDs and texts while maintaining order
        query_ids = list(self.queries.keys())
        query_texts = list(self.queries.values())

        # Extract document IDs and texts while maintaining order
        doc_ids = list(self.subsampled_corpus.keys())
        doc_texts = [self.subsampled_corpus[doc_id]['text'] for doc_id in doc_ids]

        if model is not None:
            # Encode queries
            query_encodings = model.encode(query_texts)
            # Encode documents
            doc_encodings = model.encode(doc_texts)

        

        # 3. Compute Cosine Similarity and Evaluate NDCG@10
        ndcg_scores = []

        if model is not None:
            # Iterate over queries using their indices to maintain mapping
            for idx in tqdm(range(len(query_ids)), desc="Processing Queries"):
                query_id = query_ids[idx]
                query_encoding = query_encodings[idx].reshape(1, -1)  # Reshape for cosine_similarity

                # Compute cosine similarities between the current query and all documents
                similarities = np.dot(query_encoding, doc_encodings.T)[0]

                # Get top 10 document indices based on similarity
                top_indices = np.argsort(similarities)[::-1][:10]
                
                # Retrieve the actual document IDs for the top indices
                top_doc_ids = [doc_ids[i] for i in top_indices]
                
                # Get relevance scores for the top 10 documents
                relevances = [
                    self.qrels.get(query_id, {}).get(doc_id, 0.0) for doc_id in top_doc_ids
                ]
                
                # Compute NDCG@10 for the current query using sklearn's ndcg_score
                ndcg = ndcg_score([relevances], [similarities[top_indices]], k=10)
                ndcg_scores.append(ndcg)
    
        average_ndcg = np.mean(ndcg_scores)
        return average_ndcg

