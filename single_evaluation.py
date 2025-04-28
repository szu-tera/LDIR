import logging
import os
import sys
from dataclasses import dataclass, field
import torch
from utils.sample_fun import farthest_point_sample_with_cuda
from utils.utils import gather_result_sts
from src.anchor_sim_model import cos_simi_model

import numpy as np
import mteb
import json
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import HfArgumentParser, TrainingArguments, set_seed
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments():
    """
    Arguments pertaining to which model/config/tokenizer we are going to embed.
    """
    model_name_or_path: str = field(
        default="WhereIsAI/UAE-Large-V1",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    batch_size: int = field(
        default=128
    )
    devices_num: int = field(
        default=1
    )


@dataclass
class DynamicDataEmbeddingArguments(DataTrainingArguments):
    """
    Arguments for dynamic sample of anchor test.
    """
    task_type: str = field(
        default="STS",
        metadata={"help": "task type of MTEB"}
    )

    # For filtering when using demonstrations
    data_dir: str = field(
        default=False,
        metadata={"help": "Sample set dir path"}
    )

    length_sentence_file: str = field(
        default="medi_text_100low",
        metadata={"help": ""}
    )
  
@dataclass
class DynamicSampleArguments(TrainingArguments):
    num_anchors: int = field(
        default=100
    )

    sample_type: str = field(
        default="FPS",
        metadata={"help": ""}
    )


def main():
    parser = HfArgumentParser((ModelArguments, DynamicDataEmbeddingArguments, DynamicSampleArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, sample_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, sample_args = parser.parse_args_into_dataclasses()

    # Set seed
    set_seed(sample_args.seed)

    devices_list = []
    if model_args.devices_num is not None:
        for i in range(model_args.devices_num):
            devices_list.append(f"cuda:{i}")


    anchors_dict = {}
    anchor_file = f"./data/anchors/medi_text_100low/FPS/500.json"
    corpus_emb_file = "../data/medi2_documents_dense_embedding.npy"
    if not os.path.exists(anchor_file):

        corpus_embeddings = []
        corpus = []
        if not os.path.exists(corpus_emb_file):
            logger.info("corpus embedding")
            
            with open(data_args.data_dir, "r") as f:
                corpus = json.load(f)

            embedding_model = SentenceTransformer(model_args.model_name_or_path)

            cuda_pool = embedding_model.start_multi_process_pool(target_devices=devices_list)

            corpus_embeddings = embedding_model.encode_multi_process(corpus, batch_size=model_args.batch_size, pool=cuda_pool, show_progress_bar=True)
            
            np.save(corpus_emb_file, corpus_embeddings)
        else:
            with open(data_args.data_dir, "r") as f:
                corpus = json.load(f)

            corpus_embeddings = np.load(corpus_emb_file)

        logger.info("anchor processing")
        corpus_embeddings = torch.tensor(corpus_embeddings).cuda()
        corpus_embeddings = corpus_embeddings.unsqueeze(0)
        index, selected_embedding = farthest_point_sample_with_cuda(corpus_embeddings, sample_args.num_anchors)

        selected_embedding_keys = list(selected_embedding.keys())


        for i in selected_embedding_keys:
            anchors_dict[i] = corpus[i]
        
        with open(anchor_file, "w") as f:
            json.dump(anchors_dict, f)

        logger.info("anchors have selected")
    else:
        with open(anchor_file, "r") as f:
            anchors_dict = json.load(f)
            logger.info("anchors have loaded")

    
    
    if model_args.model_name_or_path == "WhereIsAI/UAE-Large-V1":
        model_name = "UAE"
    else:
        model_name = model_args.model_name_or_path
        
    result_dir = f"results/{data_args.length_sentence_file}/{model_name}/{data_args.task_type}/{sample_args.sample_type}/{sample_args.num_anchors}"
    result_all_dir = f"results/{data_args.length_sentence_file}/{model_name}/{data_args.task_type}/{sample_args.sample_type}/{sample_args.num_anchors}/no_model_name_available/no_revision_available/a_sts_sum.json"
    result_stsb_dir = f"results/{data_args.length_sentence_file}/{model_name}/{data_args.task_type}/{sample_args.sample_type}/{sample_args.num_anchors}/no_model_name_available/no_revision_available/STSBenchmark.json"

    if not os.path.exists(result_all_dir):
        
        anchors_list = []
        anchors_dict_keys = list(anchors_dict.keys())
        for i in anchors_dict_keys:
            anchors_list.append(anchors_dict[i])
        
        if  data_args.task_type=="STS":
            tasks = mteb.get_tasks(languages=["eng"], tasks=["SICK-R","STS12","STS13","STS14","STS15","STS16","STSBenchmark"])
        
        evaluation = mteb.MTEB(tasks=tasks)
        

        encoder_model = cos_simi_model(model_args, anchors=anchors_list, batch_size=model_args.batch_size)


        evaluation.run(model=encoder_model, output_folder=result_dir)
        
        logger.info(f"start to gather single task result")

        
        if data_args.task_type == "STS":
            gather_result_sts(result_dir)

    if not os.path.exists(result_all_dir):
        logger.info(f"interpre emb have completed")
        logger.info(f"start to gather single task result")

        if data_args.task_type == "STS":
            gather_result_sts(result_dir)

if __name__ == "__main__":
    main()

