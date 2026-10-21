import numpy as np
import argparse
import time
import threading
import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score


# Function to perform inference and monitor resources
def perform_inference_with_monitoring(
    model_weight_path, dataset_path, batch_size, precision
):
    # Prepare finished flag for each batch
    finished_flag = threading.Event()
    GPU_memory_list = []

    # Function to monitor GPU memory usage
    def monitor_resources():
        while not finished_flag.is_set():
            time.sleep(1)
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # in MB
            GPU_memory_list.append(gpu_memory)

    # Load test data
    test_data_path = f"{dataset_path}/test_data.npy"
    test_label_path = f"{dataset_path}/test_label.npy"
    test_data = np.load(test_data_path)
    test_labels = np.load(test_label_path)
    actual_label_list = list(test_labels)

    test_data = torch.from_numpy(test_data).float()
    test_labels = torch.from_numpy(test_labels).long()

    # Ensure the input data is in the correct precision format
    if precision == "float16":
        test_data = test_data.half()

    # Create data loader
    test_loader = DataLoader(
        TensorDataset(test_data, test_labels), batch_size=batch_size
    )

    # Initialize resource and performance monitoring
    inference_time_list, predicted_label_list = [], []

    # Load the model
    model = torch.load(model_weight_path).to("cuda")

    # Start the monitoring thread
    monitor_thread = threading.Thread(target=monitor_resources)
    monitor_thread.start()

    # Measure throughput
    start_time = time.time()

    # Perform inference on the test dataset
    with torch.no_grad():
        for data, test_label in tqdm(test_loader):
            data = data.to("cuda")
            test_label = test_label.to("cuda")

            # Perform inference and measure latency
            inference_start_time = time.time()
            model_output = model(data)
            inference_end_time = time.time()

            # Record inference time
            inference_time_list.append(inference_end_time - inference_start_time)

            # Calculate predictions
            predictions = torch.argmax(model_output.logits, 1).cpu().numpy()
            predicted_label_list.extend(list(predictions))

    end_time = time.time()

    # Stop the monitoring thread
    finished_flag.set()
    monitor_thread.join()

    # Output maximum GPU memory usage
    maximum_gpu_memory = max(GPU_memory_list)
    print(f"Maximum GPU memory usage: {maximum_gpu_memory} MB")

    # Calculate model size
    model_size = os.path.getsize(model_weight_path) / 1024 / 1024  # in MB
    print(f"Model size in local storage: {model_size} MB")

    # Calculate average inference latency per sample
    average_inference_time = (
        sum(inference_time_list) / len(inference_time_list) / batch_size * 1000
    )
    print(f"Average inference latency per sample: {average_inference_time} ms")

    # Calculate throughput
    throughput_per_second = len(test_data) / (end_time - start_time)
    print(f"Throughput per second: {throughput_per_second} samples per second")

    # Calculate classification accuracy
    correct_predictions = np.sum(
        np.array(predicted_label_list) == np.array(actual_label_list)
    )
    accuracy = correct_predictions / len(test_data)
    print(f"Classification accuracy: {accuracy}")

    # Calculate precision
    precision = precision_score(
        actual_label_list, predicted_label_list, average="macro"
    )
    print(f"Precision: {precision}")

    # Calculate recall
    recall_rate = recall_score(actual_label_list, predicted_label_list, average="macro")
    print(f"Recall rate: {recall_rate}")

    # Calculate F1-score
    f1 = f1_score(actual_label_list, predicted_label_list, average="macro")
    print(f"F1-score: {f1}")


def main():
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="ONNX Model Inference")
    parser.add_argument(
        "--model_weight_path",
        type=str,
        required=True,
        help="Path to the ONNX model weight file",
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to the dataset directory"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for inference"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="float32",
        help="Precision of the input data (e.g., float32)",
    )

    args = parser.parse_args()
    model_weight_path = args.model_weight_path
    dataset_path = args.dataset_path
    batch_size = args.batch_size
    precision = args.precision

    perform_inference_with_monitoring(
        model_weight_path, dataset_path, batch_size, precision
    )


if __name__ == "__main__":
    main()
