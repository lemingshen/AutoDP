import prompt, tool_list

from langgraph.graph import END
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

code_reference_database = Chroma(
    collection_name="code_reference",
    embedding_function=tool_list.embeddings,
    persist_directory="./chroma_langchain_db",
)
code_reference_graph_builder = StateGraph(MessagesState)

code_reference_database.add_documents(
    [
        Document(
            metadata={
                "description": "A sample Python script to monitor the CPU memory usage."
            },
            page_content="""```python\nimport os
import time
import psutil
import threading
import multiprocessing

from tqdm import tqdm


def model_inference():
    finished = False
    CPU_memory_list = []
    handle = psutil.Process(os.getpid())

    # CPU memory monitoring function
    def monitor_resources():
        while not finished:
            time.sleep(1)
            cpu_memory = handle.memory_info().rss / 1024 / 1024  # in MB
            CPU_memory_list.append(cpu_memory)

    # Start the monitoring thread
    thread = threading.Thread(target=monitor_resources)
    thread.start()

    # create model, data loader, etc.
    # model = None
    # data_loader = None

    # You may deploy the model here
    # Model deployment, inference, or any other task ...
    # for data, label in tqdm(data_loader):
    # ......

    # Stop the monitoring thread when the work is done
    finished = True
    thread.join()
    maximum_cpu_memory = max(CPU_memory_list)
    print(f"Maximum CPU memory usage: {maximum_cpu_memory} MB")


if __name__ == "__main__":
    # Create a new process and perform model inference independently
    process = multiprocessing.Process(target=model_inference)
    process.start()
    process.join()
```""",
        ),
        Document(
            metadata={
                "description": "A sample Python script to monitor the GPU memory usage."
            },
            page_content="""```python\nimport time
import torch
import threading
import multiprocessing

from tqdm import tqdm


def model_inference():
    finished = False
    GPU_memory_list = []

    # GPU memory monitoring function
    def monitor_resources():
        while not finished:
            time.sleep(1)
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # in MB
            GPU_memory_list.append(gpu_memory)

    # Start the monitoring thread
    thread = threading.Thread(target=monitor_resources)
    thread.start()

    # create model, data loader, etc.
    # NOTE: CUDA related operations should only be included in this process
    # model = None
    # data_loader = None

    # You may deploy the model here
    # Model deployment, inference, or any other task ...
    # for data, label in tqdm(data_loader):
    # ......

    # Stop the monitoring thread when the work is done
    finished = True
    thread.join()
    maximum_gpu_memory = max(GPU_memory_list)
    print(f"Maximum GPU memory usage: {maximum_gpu_memory} MB")


if __name__ == "__main__":
    # Create a new process and perform model inference independently
    process = multiprocessing.Process(target=model_inference)
    process.start()
    process.join()
```""",
        ),
        Document(
            metadata={
                "description": "A sample Python script to measure the model size by calculating the disk space occupied by the model's parameter file."
            },
            page_content="""```python\nimport os
import torch

# suppose this is the machine learning model
model = None
torch.save(model, "model.pth")
model_size = os.path.getsize("model.pth") / 1024 / 1024  # in MB
print(f"Model size in local storage: {model_size} MB")
```""",
        ),
        Document(
            metadata={
                "description": "A sample Python script to measure the inference latency."
            },
            page_content="""```python\nimport time
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inference_time_list = []

# Suppose the test_loader contains all the test data
with torch.no_grad():
    for test_data, test_label in test_loader:
        test_data, test_label = test_data.to(device), test_label.to(device)
        batch_size = test_data.size(0)

        # Perform inference on the test data
        start_time = time.time()
        output = model(test_data)
        end_time = time.time()

# Inference latency per sample
average_inference_time = sum(inference_time_list) / len(inference_time_list) / batch_size * 1000
print(f"Average inference latency per sample: {average_inference_time} ms")
```""",
        ),
        Document(
            metadata={
                "description": "A sample Python script to measure the throughput."
            },
            page_content="""```python\nimport time
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inference_time_list = []

# Suppose the test_loader contains all the test data
start_time = time.time()
with torch.no_grad():
    for test_data, test_label in test_loader:
        test_data, test_label = test_data.to(device), test_label.to(device)
        batch_size = test_data.size(0)

        # Perform inference on the test data
        output = model(test_data)
        _, predicted = torch.max(output, 1)

end_time = time.time()

# Throughput per second
throughput_per_second = len(test_loader.dataset) / (end_time - start_time) * batch_size
print(f"Throughput per second: {throughput_per_second} samples per second")
```""",
        ),
        Document(
            metadata={
                "description": "A sample Python script to compute the classification accuracy."
            },
            page_content="""```python\nimport torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Suppose the test_loader contains all the test data
correct, total = 0, 0
with torch.no_grad():
    for test_data, test_label in test_loader:
        test_data, test_label = test_data.to(device), test_label.to(device)

        # Perform inference on the test data
        output = model(test_data)
        _, predicted = torch.max(output.data, 1)
        total += test_label.size(0)
        correct += (predicted == test_label).sum().item()

accuracy = correct / total
print(f"Classification accuracy: {accuracy}")
```""",
        ),
        Document(
            metadata={
                "description": "A sample Python script to compute the precision."
            },
            page_content="""```python\nimport torch
from sklearn.metrics import precision_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Suppose the test_loader contains all the test data
actual_label_list, predicted_label_list = [], []
with torch.no_grad():
    for test_data, test_label in test_loader:
        test_data, test_label = test_data.to(device), test_label.to(device)
        actual_label_list.extend(list(test_label.cpu().numpy()))

        # Perform inference on the test data
        output = model(test_data)
        _, predicted = torch.max(output.data, 1)
        predicted_label_list.extend(list(predicted.cpu().numpy()))

precision = precision_score(actual_label_list, predicted_label_list, average="macro")
print(f"Precision: {precision}")
```""",
        ),
        Document(
            metadata={
                "description": "A sample Python script to compute the recall rate."
            },
            page_content="""```python\nimport torch
from sklearn.metrics import recall_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Suppose the test_loader contains all the test data
actual_label_list, predicted_label_list = [], []
with torch.no_grad():
    for test_data, test_label in test_loader:
        test_data, test_label = test_data.to(device), test_label.to(device)
        actual_label_list.extend(list(test_label.cpu().numpy()))

        # Perform inference on the test data
        output = model(test_data)
        _, predicted = torch.max(output.data, 1)
        predicted_label_list.extend(list(predicted.cpu().numpy()))

recall_rate = recall_score(actual_label_list, predicted_label_list, average="macro")
print(f"Recall rate: {recall_rate}")
```""",
        ),
        Document(
            metadata={"description": "A sample Python script to compute the F1-score."},
            page_content="""```python\nimport torch
from sklearn.metrics import f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Suppose the test_loader contains all the test data
actual_label_list, predicted_label_list = [], []
with torch.no_grad():
    for test_data, test_label in test_loader:
        test_data, test_label = test_data.to(device), test_label.to(device)
        actual_label_list.extend(list(test_label.cpu().numpy()))

        # Perform inference on the test data
        output = model(test_data)
        _, predicted = torch.max(output.data, 1)
        predicted_label_list.extend(list(predicted.cpu().numpy()))

f1 = f1_score(actual_label_list, predicted_label_list, average="macro")
print(f"F1-score: {f1}")
```""",
        ),
        Document(
            metadata={"description": "A sample Python script to compute the MAE."},
            page_content="""```python\nimport torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mae_loss_list = []


def mae_loss(y_pred, y_true):
    return torch.mean(torch.abs(y_pred - y_true))


# Suppose the test_loader contains all the test data
with torch.no_grad():
    for test_data, test_label in test_loader:
        test_data, test_label = test_data.to(device), test_label.to(device)

        # Perform inference on the test data
        output = model(test_data)
        mae_loss_list.append(float(mae_loss(output, test_label).item()))


average_mae_loss = sum(mae_loss_list) / len(mae_loss_list)
print(f"Mean Absolute Error: {average_mae_loss}")
```""",
        ),
        Document(
            metadata={"description": "A sample Python script to compute the MSE."},
            page_content="""```python\nimport torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mse_loss_function = torch.nn.MSELoss()
mse_loss_list = []


# Suppose the test_loader contains all the test data
with torch.no_grad():
    for test_data, test_label in test_loader:
        test_data, test_label = test_data.to(device), test_label.to(device)

        # Perform inference on the test data
        output = model(test_data)
        mse_loss_list.append(float(mse_loss_function(output, test_label).item()))


average_mse_loss = sum(mse_loss_list) / len(mse_loss_list)
print(f"Mean Squared Error: {average_mse_loss}")
```""",
        ),
        Document(
            metadata={"description": "A sample Python script to compute the RMSE."},
            page_content="""```python\nimport torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rmse_loss_list = []


def rmse_loss(y_pred, y_true):
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2))


# Suppose the test_loader contains all the test data
with torch.no_grad():
    for test_data, test_label in test_loader:
        test_data, test_label = test_data.to(device), test_label.to(device)

        # Perform inference on the test data
        output = model(test_data)
        rmse_loss_list.append(float(rmse_loss(output, test_label).item()))


average_mse_loss = sum(rmse_loss_list) / len(rmse_loss_list)
print(f"Root Mean Squared Error: {average_mse_loss}")
```""",
        ),
        Document(
            metadata={
                "description": "A sample Python script to compute the R-squared."
            },
            page_content="""```python\nimport torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
r2_loss_list = []


def r_squared_loss(y_true, y_pred):
    \"\"\"
    Calculate the R-squared (coefficient of determination) for a regression model using PyTorch.

    Args:
        y_true (torch.Tensor): Tensor of true target values.
        y_pred (torch.Tensor): Tensor of predicted target values.

    Returns:
        torch.Tensor: R-squared value as a scalar tensor.
    \"\"\"
    # Ensure inputs are PyTorch tensors
    y_true = y_true if isinstance(y_true, torch.Tensor) else torch.tensor(y_true)
    y_pred = y_pred if isinstance(y_pred, torch.Tensor) else torch.tensor(y_pred)

    # Mean of true values
    y_mean = torch.mean(y_true)

    # Residual sum of squares (SS_res)
    ss_res = torch.sum((y_true - y_pred) ** 2)

    # Total sum of squares (SS_tot)
    ss_tot = torch.sum((y_true - y_mean) ** 2)

    # Calculate R-squared
    r2 = 1 - (ss_res / ss_tot)

    return r2


# Suppose the test_loader contains all the test data
with torch.no_grad():
    for test_data, test_label in test_loader:
        test_data, test_label = test_data.to(device), test_label.to(device)

        # Perform inference on the test data
        output = model(test_data)
        r2_loss_list.append(float(r_squared_loss(output, test_label).item()))


average_r2 = sum(r2_loss_list) / len(r2_loss_list)
print(f"R-squared: {average_r2}")
```""",
        ),
        Document(
            metadata={"description": "A sample Python script to compute the MAPE."},
            page_content="""```python\nimport torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mape_loss_list = []


def mape_loss(y_true, y_pred):
    \"\"\"
    Calculate the Mean Absolute Percentage Error (MAPE) loss.

    Args:
        y_true (torch.Tensor): Tensor of true target values.
        y_pred (torch.Tensor): Tensor of predicted target values.

    Returns:
        torch.Tensor: MAPE loss as a scalar tensor.
    \"\"\"
    # Ensure inputs are PyTorch tensors
    y_true = y_true if isinstance(y_true, torch.Tensor) else torch.tensor(y_true)
    y_pred = y_pred if isinstance(y_pred, torch.Tensor) else torch.tensor(y_pred)

    # Calculate MAPE
    epsilon = 1e-8  # Add a small epsilon to avoid division by zero
    loss = torch.mean(torch.abs((y_true - y_pred) / (y_true + epsilon))) * 100

    return loss


# Suppose the test_loader contains all the test data
with torch.no_grad():
    for test_data, test_label in test_loader:
        test_data, test_label = test_data.to(device), test_label.to(device)

        # Perform inference on the test data
        output = model(test_data)
        mape_loss_list.append(float(mape_loss(output, test_label).item()))


average_mape = sum(mape_loss_list) / len(mape_loss_list)
print(f"Mean Absolute Percentage Error: {average_mape}")
```""",
        ),
        Document(
            metadata={
                "description": "A sample Python script to compute the BLEU score."
            },
            page_content="""```python\nfrom nltk.translate.bleu_score import sentence_bleu

reference_sentence = [["this", "is", "a", "test"], ["this", "is" "test"]]
model_output = ["this", "is", "a", "test", "for", "bleu", "score"]

blue_score = score = sentence_bleu(reference_sentence, model_output)
print(f"BLEU score: {blue_score}")
```""",
        ),
        Document(
            metadata={
                "description": "A sample Python script to compute the ROUGE score."
            },
            page_content="""```python\nfrom rouge import Rouge

reference_sentence = "This is a test for rouge score."
model_output = "This is a test."

rouge = Rouge()
rouge_score = rouge.get_scores(hyps=model_output, refs=reference_sentence)
rouge_1_recall = rouge_score[0]["rouge-1"]["r"]
rouge_1_precision = rouge_score[0]["rouge-1"]["p"]
rouge_1_f1 = rouge_score[0]["rouge-1"]["f"]
rouge_2_recall = rouge_score[0]["rouge-2"]["r"]
rouge_2_precision = rouge_score[0]["rouge-2"]["p"]
rouge_2_f1 = rouge_score[0]["rouge-2"]["f"]
rouge_l_recall = rouge_score[0]["rouge-l"]["r"]
rouge_l_precision = rouge_score[0]["rouge-l"]["p"]
rouge_l_f1 = rouge_score[0]["rouge-l"]["f"]

print(f"ROUGE-1 Recall: {rouge_1_recall}")
print(f"ROUGE-1 Precision: {rouge_1_precision}")
print(f"ROUGE-1 F1 Score: {rouge_1_f1}")
print(f"ROUGE-2 Recall: {rouge_2_recall}")
print(f"ROUGE-2 Precision: {rouge_2_precision}")
print(f"ROUGE-2 F1 Score: {rouge_2_f1}")
print(f"ROUGE-L Recall: {rouge_l_recall}")
print(f"ROUGE-L Precision: {rouge_l_precision}")
print(f"ROUGE-L F1 Score: {rouge_l_f1}")
```""",
        ),
    ]
)


@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = code_reference_database.similarity_search(query, k=10)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


# step 1: generate an AIMessage that may include a tool-call to be sent
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = tool_list.model.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])

    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}


# Step 2: Execute the retrieval.
tools = ToolNode([retrieve])


# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    retrieved_context = "\n\n".join(document.content for document in tool_messages)
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    response_generation_prompt = [
        SystemMessage(prompt.code_reference_system_prompt.format(retrieved_context))
    ] + conversation_messages

    # Run
    response = tool_list.model.invoke(response_generation_prompt)
    return {"messages": [response]}


code_reference_graph_builder.add_node(query_or_respond)
code_reference_graph_builder.add_node(tools)
code_reference_graph_builder.add_node(generate)

code_reference_graph_builder.set_entry_point("query_or_respond")
code_reference_graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
code_reference_graph_builder.add_edge("tools", "generate")
code_reference_graph_builder.add_edge("generate", END)
code_reference_graph = code_reference_graph_builder.compile()
