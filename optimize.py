from transformers import AutoModelForAudioClassification
import torch
import numpy as np
import torch_pruning as tp
import multiprocessing
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import os
import threading
import time


def inference_process(result_queue, batch_size, precision):
    finished = threading.Event()
    GPU_memory_list = []
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_max_memory_cached()
    model = torch.load("model.pth")
    model = model.to("cuda")

    def monitor_resources():
        while not finished.is_set():
            time.sleep(1)
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # in MB
            GPU_memory_list.append(gpu_memory)

    # Start resource monitoring
    thread = threading.Thread(target=monitor_resources)
    thread.start()

    # Create data loader
    test_data = np.load("./dataset/Whisper/test_data.npy")
    test_labels = np.load("./dataset/Whisper/test_label.npy")
    test_data = torch.from_numpy(test_data).float()
    if precision == "float16":
        test_data = test_data.half()
    test_labels = torch.from_numpy(test_labels).long()
    data_loader = DataLoader(
        TensorDataset(test_data, test_labels), batch_size=batch_size, shuffle=False
    )

    # Evaluate the model
    correct = 0
    total = 0
    count = 0
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader):
            inputs, labels = inputs.to("cuda"), labels.to("cuda")  # Move data to GPU
            outputs = model(inputs)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            count += 1
            if count == 10:
                break

    finished.set()
    thread.join()

    maximum_gpu_memory = max(GPU_memory_list)
    result_queue.put(maximum_gpu_memory)


def objective(trial: optuna.Trial):
    try:
        # model precision
        precision = trial.suggest_categorical("precision", ["float32", "float16"])
        # Batch size tuning
        batch_size = trial.suggest_categorical("batch_size", [1, 2, 4, 8, 12, 16])
        # Pruning rate tuning
        pruning_ratio = trial.suggest_categorical(
            "pruning_ratio",
            [
                0.05,
                0.1,
                0.15,
                0.2,
                0.25,
                0.3,
                0.35,
                0.4,
                0.45,
                0.5,
                0.125,
                0.0625,
                0.03125,
                0.015625,
            ],
        )

        # Load model
        model = AutoModelForAudioClassification.from_pretrained(model_id)
        model = model.to("cuda")  # Move the model to GPU
        model.eval()

        # Perform model pruning
        # Define importance criterion: L2 Norm of grouped weights
        imp = tp.importance.GroupNormImportance(p=2)
        example_inputs = torch.randn(1, 128, 3000).to("cuda")
        feature_dim = 1280

        # Identify layers to be ignored during pruning
        ignored_layers = []
        for m in model.modules():
            if isinstance(m, torch.nn.Linear) and m.out_features == 7:
                ignored_layers.append(m)  # DO NOT prune the final classifier!

        # Initialize pruner
        pruner = tp.pruner.MetaPruner(
            model,
            example_inputs,
            importance=imp,
            pruning_ratio=pruning_ratio,
            ignored_layers=ignored_layers,
            round_to=8,
        )

        # Prune the model
        pruner.step()

        # Prune the positional embeddings
        embedding_weight = model.encoder.embed_positions.weight.data
        pruned_number = int(feature_dim * pruning_ratio)
        sum_list = [
            torch.abs(embedding_weight[:, index]).sum() for index in range(feature_dim)
        ]
        indexed_weights = list(enumerate(sum_list))
        sorted_weights = sorted(indexed_weights, key=lambda x: x[1])
        index_list = [index for index, _ in sorted_weights[:pruned_number]]

        tp.prune_embedding_out_channels(
            model.encoder.embed_positions,
            idxs=index_list,
        )

        # Modify the number of heads and head_dim of the attention layers
        for layer_index in range(32):
            model.encoder.layers[layer_index].self_attn.embed_dim = int(
                model.encoder.layers[layer_index].self_attn.embed_dim
                * (1 - pruning_ratio)
            )
            if pruning_ratio in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
                model.encoder.layers[layer_index].self_attn.num_heads = int(
                    model.encoder.layers[layer_index].self_attn.num_heads
                    * (1 - pruning_ratio)
                )
            else:
                model.encoder.layers[layer_index].self_attn.head_dim = int(
                    model.encoder.layers[layer_index].self_attn.head_dim
                    * (1 - pruning_ratio)
                )

        # Adopt 'float16' precision
        if precision == "float16":
            model = model.half()

        # save the model and clear the cache
        torch.save(model, "model.pth")
        model_size = os.path.getsize("model.pth") / 1024 / 1024  # in MB

        # Start model inference process
        result_queue = multiprocessing.Queue()
        model_inference_process = multiprocessing.Process(
            target=inference_process, args=(result_queue, batch_size, precision)
        )

        model_inference_process.start()
        model_inference_process.join()

        maximum_gpu_memory = result_queue.get()
    except Exception as e:
        print(f"Error: {e}")
        maximum_gpu_memory = 1000000
        model_size = 1000000

    return maximum_gpu_memory, model_size


if __name__ == "__main__":
    model_id = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
    sampler = TPESampler()  # Use TPE sampler for more advanced hyperparameter searching
    pruner = MedianPruner()  # Use Median Pruner for more efficient pruning
    study = optuna.create_study(
        directions=["minimize", "minimize"], sampler=sampler, pruner=pruner
    )
    study.optimize(objective, n_trials=100)  # Increased number of trials to 50

    print("Trials with all resource requirements satisfied:")
    for trial in study.trials:
        max_gpu_mem, model_size = trial.values
        if max_gpu_mem == 1000000 and model_size == 1000000:
            print(f"Error Trial: {trial.params}")
        if max_gpu_mem < 2400 and model_size < 2400:  # Check resource constraints
            print(
                f"Current Trial: Model Size: {model_size} MB, GPU Memory Usage: {max_gpu_mem} MB"
            )
            print(f"Satisfactory Trial: {trial.params}")
