import os
import tool_list

from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_community.document_loaders.firecrawl import FireCrawlLoader

model_optimization_reference_database = Chroma(
    collection_name="model_optimization_sample_code",
    embedding_function=tool_list.embeddings,
    persist_directory="./chroma_langchain_db",
)

document_list = [
    Document(
        metadata={
            "description": "Sample code for model pruning using `torch_pruning` for PyTorch models.",
            "source": "https://github.com/VainF/Torch-Pruning",
        },
        page_content="""import torch
import numpy as np
from torchvision.models import resnet18
import torch_pruning as tp

model = resnet18()
class_number = 1000
# example input for the model
# the example input must be loaded from the local file
example_inputs = np.load('sample_data.npy')
example_inputs = torch.from_numpy(example_inputs).float()

# 1. Importance criterion, here we calculate the L2 Norm of grouped weights as the importance score
imp = tp.importance.GroupNormImportance(p=2)

# 2. Initialize a pruner with the model and the importance criterion
ignored_layers = []
for m in model.modules():
    if isinstance(m, torch.nn.Linear) and m.out_features == class_number:
        ignored_layers.append(m)  # DO NOT prune the final classifier!

pruner = tp.pruner.MetaPruner(
    model,
    example_inputs,
    importance=imp,
    pruning_ratio=0.5,  # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
    ignored_layers=ignored_layers,
    round_to=8,
)

# 3. Prune the model
pruner.step()
""",
    ),
    Document(
        metadata={"description": "Sample code for model quantization using PyTorch"},
        page_content="""import torch
import torch.ao.quantization
from torchvision.models import resnet18
from tqdm import tqdm

import time
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Load the test data and labels
test_data = np.load("./dataset/ILSVRC/test_data.npy")
test_labels = np.load("./dataset/ILSVRC/test_label.npy")

# Convert to PyTorch tensors
test_data_tensor = torch.tensor(test_data, dtype=torch.float32)
test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

# Create a DataLoader
test_dataset = TensorDataset(test_data_tensor, test_labels_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = resnet18()
model.eval()

# Perform model quantization
# Only qint8 is supported for dynamic quantization
# The quantized model can only be deployed on CPU
quantized_model = torch.ao.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

torch.save(quantized_model.state_dict(), "model.pth")
print(os.path.getsize("model.pth") / 1024 / 1024)

# Perform inference with the quantized model
start_time = time.time()
for images, labels in tqdm(test_loader):
    quantized_model(images)
end_time = time.time()
print(f"Elapsed time: {end_time - start_time}")
""",
    ),
]

model_optimization_reference_database.add_documents(documents=document_list)


@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = model_optimization_reference_database.similarity_search(query, k=5)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )

    print("Retrieved the following information from the database:")
    print("\033[33m{}\033[0m".format(retrieved_docs))

    return serialized, retrieved_docs
