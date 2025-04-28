from src.anchor_sim_model import cos_simi_model
import json
import os

dirname = os.path.dirname(__file__)

anchor_file = os.path.join(dirname, "./data/medi_text_100low/FPS/500.json")
with open(anchor_file, "r") as f:
    anchors_dict = json.load(f)

anchors_list = []
anchors_dict_keys = list(anchors_dict.keys())
for i in anchors_dict_keys:
    anchors_list.append(anchors_dict[i])

encoder_model = cos_simi_model("WhereIsAI/UAE-Large-V1", anchors=anchors_list)

documents = [
    "This is a test document", 
    "This is another test document"
]
# Perform inference
embeddings = encoder_model.encode(documents) 