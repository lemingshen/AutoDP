from transformers import AutoModelForAudioClassification
import torch
import numpy as np
import multiprocessing
from torch.utils.data import DataLoader, TensorDataset
import os
import threading
import time
from tqdm import tqdm

model_id = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"


def load_model_from_file():
    model = AutoModelForAudioClassification.from_pretrained(model_id)
    model.to("cuda")  # Move the model to GPU
    return model


def load_test_data():
    test_data = np.load("./dataset/Whisper/test_data.npy")
    test_labels = np.load("./dataset/Whisper/test_label.npy")
    return test_data, test_labels


def create_data_loader(test_data, test_labels, batch_size=16):
    test_tensor = torch.tensor(test_data, dtype=torch.float32)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)
    dataset = TensorDataset(test_tensor, test_labels_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loader


def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader):
            inputs, labels = inputs.to("cuda"), labels.to("cuda")  # Move data to GPU
            outputs = model(inputs)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f"Accuracy: {accuracy * 100:.2f}%")


def inference_process():
    finished = False
    gpu_memory_list = []

    def monitor_resources():
        while not finished:
            time.sleep(1)
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # in MB
            gpu_memory_list.append(gpu_memory)

    # Start the monitoring thread
    thread = threading.Thread(target=monitor_resources)
    thread.start()

    model = load_model_from_file()

    # Monitor model size on disk
    torch.save(model, "whisper-large-v3.pth")
    model_size = os.path.getsize("whisper-large-v3.pth") / 1024 / 1024  # in MB
    print(f"Model size in local storage: {model_size} MB")

    test_data, test_labels = load_test_data()
    data_loader = create_data_loader(test_data, test_labels)

    evaluate_model(model, data_loader)

    # Stop the monitoring thread when the work is done
    finished = True
    thread.join()
    maximum_gpu_memory = max(gpu_memory_list)
    print(f"Maximum GPU memory usage: {maximum_gpu_memory} MB")


if __name__ == "__main__":
    process = multiprocessing.Process(target=inference_process)
    process.start()
    process.join()
