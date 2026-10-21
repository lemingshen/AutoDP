from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

# %% prompt for device specs searching
device_spec_example_list = [
    {
        "input": "NVIDIA Jetson Nano",
        "output": """1. Processor (CPU)
- Architecture: Quad-core ARM Cortex-A57
- Clock Speed: 1.43 GHz
2. Graphics Processing Unit (GPU)
- Model: 128-core NVIDIA Maxwell architecture
- Base Frequency: 640 MHz
- Boost Frequency: Up to 921 MHz
3. Memory (RAM)
- Size: 4 GB
- Type: 64-bit LPDDR4
- Bandwidth: 25.6 GB/s
4. Storage
- Internal Storage: 16 GB eMMC 5.1
- Expandable Storage: microSD card slot (card not included)
5. Connectivity
- Networking: Gigabit Ethernet (10/100/1000BASE-T)
- Wireless: Optional via M.2 Key E slot
6. Input/Output (I/O) Ports
- USB Ports:
  - 1x USB 3.0 Type-A
  - 3x USB 2.0
- Display Outputs:
  - HDMI 2.0
  - eDP 1.4
- Camera Interface: 1x MIPI CSI-2
- Others:
  - GPIO
  - I2C
  - I2S
  - SPI
  - UART
7. Power
- Input Voltage: 5V
- Power Consumption: Configurable between 5W and 10W
- Power Options:
  - Micro-USB 5V 2A
  - DC power adapter 5V 4A
8. Operating System
- Supported OS: Linux-based distributions
- SDK: NVIDIA JetPack SDK, which includes support for CUDA, cuDNN, and TensorRT""",
    },
    {
        "input": "NVIDIA Jetson TX2",
        "output": """1. Processor (CPU)
- Architecture: Dual-core NVIDIA Denver 2 64-bit CPU and quad-core ARM Cortex-A57 MPCore processor
- Max Frequency: Denver 2: Up to 2.0 GHz; Cortex-A57: Up to 2.0 GHz
2. Graphics Processing Unit (GPU)
- Model: 256-core NVIDIA Pascal architecture GPU
- Max Frequency: Up to 1.3 GHz
3. Memory (RAM)
- Size: 8 GB
- Type: 128-bit LPDDR4
- Bandwidth: Up to 59.7 GB/s
4. Storage
- Internal Storage: 32 GB eMMC 5.1
5. Connectivity
- Networking: Gigabit Ethernet (10/100/1000BASE-T)
- Wireless: Optional via M.2 Key E slot
6. Input/Output (I/O) Ports
- USB Ports:
  - Up to 3x USB 3.0 (5 Gbps)
  - 3x USB 2.0
- Display Outputs:
  - 2x multi-mode DP 1.2/eDP 1.4/HDMI 2.0
  - 2x4 DSI (1.5 Gbps/lane)
- Camera Interface: Up to 6 cameras (12 via virtual channels) with 12 lanes MIPI CSI-2 D-PHY 1.2 (up to 30 Gbps)
- Other Interfaces:
  - 5x UART
  - 3x SPI
  - 4x I2S
  - 8x I2C
  - 2x CAN
  - GPIOs
7. Power
- Input Voltage: 5.5V to 19.6V DC
- Power Consumption: Configurable between 7.5W and 15W
8. Operating System
- Supported OS: Linux-based distributions
- SDK: NVIDIA JetPack SDK, which includes support for CUDA, cuDNN, and TensorRT""",
    },
]
device_spec_example_prompt = ChatPromptTemplate.from_messages(
    [("human", "Please search for detailed specs of {input}."), ("ai", "{output}")]
)
device_spec_few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=device_spec_example_list, example_prompt=device_spec_example_prompt
)
device_spec_final_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI assistant that can use the web search tool to search for information. Please search for detailed specs of an device based on the user instruction. The specs should include detailed information about CPU, GPU, RAM, Storage, Connectivity, Input/Output Ports, Power, and Operating System. Please directly output the specs in Markdown format without additional contents.",
        ),
        device_spec_few_shot_prompt,
        ("human", "Please search for detailed specs of {input}"),
        ("placeholder", "{searched_results}"),
    ]
)


# %% prompt for dataset information searching
dataset_information_searching_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI assistant that can use the web search tool to search for information. Please search for detailed information of the dataset specified by the user, including but not limited to the dataset size, the dimensionality of the data sample, the number of classes, and some Python code to download the dataset, etc.",
        ),
        (
            "human",
            "Please search for detailed information of the {input} dataset. Then, please extract all the URLs you've searched and output each URL on a new line. Please directly output the URLs without extra contents.",
        ),
        ("placeholder", "{searched_results}"),
    ]
)


# %% prompt for requirement standardization
requirement_standardization_example_list = [
    {
        "input": "To deploy a ResNet-152 model on Jetson Nano, the model's CPU and GPU memory usage should stay below 70% and 80% of available resources, respectively, with a total model size under 100 MB. Inference latency must be ≤ 50 ms, with power consumption capped at 10 watts. The system should process at least 20 images per second while maintaining a top-1 accuracy of ≥ 75%, precision of ≥ 80%, recall of ≥ 85%, F1-score of ≥ 80%, and specificity of ≥ 90%. These requirements ensure efficient deployment with robust performance.",
        "output": "1. CPU memory: less than 70% of available resources\n2. GPU memory: less than 80% of available resources\n3. Model size: less than 100 MB of disk space\n4. Inference latency: less then or equal to 50 ms\n5. Throughput: at least 20 images per second\n6. Classification accuracy: greater than or equal to 75%\n7. Precision: at least 80%\n8. F1-score: NA\n9. Specificity: NA",
    },
    {
        "input": "To deploy LLaMA3-8B on a Jetson TX2, the model’s CPU memory usage must remain under 50% of the 8 GB RAM, with GPU memory limited to 5 GB. The model size, including weights and dependencies, should not exceed 32 GB, ideally compressed to below 10 GB. Inference latency must be ≤ 200 ms per token, with a throughput of at least 5 tokens/second. Performance metrics should include perplexity ≤ 15, BLEU score ≥ 30 for translation, and ROUGE-L score ≥ 40 for summarization.",
        "output": "1. CPU memory: less than the available memory\n2. GPU memory: less than the available memory\n3. Model size: model parameters takes up less than 10 GB of disk space\n4. Inference latency: less than 200 ms per token\n5. Throughput: at least 5 tokens/second\n6. Perplexity: less then 15\n7. BLEU score: great than 30\n8. ROUGE score: NA",
    },
]
requirement_standardization_example_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "Please format the following requirements:\n{input}",
        ),
        ("ai", "{output}"),
    ]
)
requirement_standardization_few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=requirement_standardization_example_list,
    example_prompt=requirement_standardization_example_prompt,
)

requirement_standardization_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a professional document formatter. Given the user requirements, please format them into Markdown format with each requirement on a new line. Please directly output the formatted requirements. Some examples will be provided for your reference.",
        ),
        requirement_standardization_few_shot_prompt,
        (
            "human",
            "Please format the following requirements:\n{input}",
        ),
    ]
)

# %% requirement splitting
resource_requirement_extraction_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a professional AI assistant. Given the user requirements, please extract the resource-related requirements (e.g., CPU memory usage, GPU memory usage, and model size). Please directly output the extracted requirements with each one in one line.",
        ),
        (
            "human",
            "Please extract the resource-related requirements from the following user requirements:\n{input}",
        ),
    ]
)

performance_requirement_extraction_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a professional AI assistant. Given the user requirements, please extract the performance-related requirements (e.g., inference latency, throughput, accuracy, etc.). Please directly output the extracted requirements with each one in one line.",
        ),
        (
            "human",
            "Please extract the performance-related requirements from the following user requirements:\n{input}",
        ),
    ]
)

# %% model information retrieval prompt
model_information_system_prompt = "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know. Directly output the answer without extra contents and keep the answer concise.\n\n<context>\n{}\n</context>"

model_name_extraction_prompt = "Based on the information provided in the database, please directly output the name of the model without extra content."

model_information_retrieval_prompt = "You must retrieve relevant and detailed information from the database to implement my request. Please tell me how to use the {} model using Python. Please directly output the code without extra contents."

reinvoke_retrieval_prompt = "Based on the retrieved information, please judge whether the user's query can be answered. If not, you must modify the query and invoke the retrieval tool again to retrieve different but relevant information from the database to answer the user's query."

# %% code reference retrieval prompt
code_reference_system_prompt = "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know. Directly output the answer without extra contents and keep the answer concise.\n\n<context>\n{}\n</context>"


# %% Task 1: Model profiling via local deployment
### Task 1.1: generate code for model running on local device
model_deployment_code_prompt = {
    "messages": [
        {
            "role": "system",
            "content": "You are a professional Python programmer, especially in the field of machine learning. Based on the sample code for deploying the original {} model from Hugging Face, you are tasked with optimizing the model due to limited CPU, GPU, and disk storage resources. You may use the web search tool if necessary.\n\n<sample_deployment_code>\n```python\n{}\n```\n</sample_deployment_code>",
        },
        {
            "role": "user",
            "content": "Let's think step by step. First, based on the sample code provided by Hugging Face, please write a Python script to deploy the model disregarding the resource constraints. Note that the model should be executed on {}. Please use the `multiprocessing` package to create an independent process to perform model inference. Please directly output the code without additional contents.",
        },
    ]
}

### Task 1.2: modify the code by adding local testing dataset
model_deployment_on_local_test_dataset_prompt = {
    "messages": [
        {
            "role": "user",
            "content": """Second, please modify the code by changing the dataset to a local testing dataset. Please consider the following steps:\n1. Load the test data and labels from the local path '{}'. The test data is stored in a numpy array named 'test_data.npy' and the test labels are stored in a numpy array named 'test_label.npy'.\n2. Pack the data and labels into a DataLoader object.\n3. Modify the code by evaluating the model on the local dataset.\nPlease directly output the modified code without extra contents. Below are some additional information of the local dataset:\n\n<local_dataset_info>\n{}\n</local_dataset_info>""",
        }
    ]
}


### Task 1.3: Generate code for model monitoring on local device
#### Task 1.3.1: retrieve relevant reference code
resource_monitoring_code_reference_prompt = "Based on the user requirements listed below, please first analyze which part is related to resource monitoring (e.g., CPU, GPU, and model size). Then, based on the information provided in the database, please retrieve relevant code snippets according to the resource monitoring part of the user requirements. Please do not provide reference code that is not specified by the user requirements listed below. You should adopt the following format for the final output:\n1. Reference code for <item1> monitoring\n```python\n<code>\n```\n2. Reference code for <item2>\n```python\n<code>\n```\n......\n\n<user_requirements>\n{}\n</user_requirements>"

performance_monitoring_code_reference_prompt = "Based on the user requirements listed below, please retrieve all relevant code snippets from the database for model performance monitoring. Please do not provide reference code that is not specified by the user requirements listed below. You shoud adopt the following format for the final output:\n1. Reference code for <item1> monitoring\n```python\n<code>\n```\n2. Reference code for <item2>\n```python\n<code>\n```\n......\n\n<user_requirements>\n{}\n</user_requirements>"


#### Task 1.3.2: generate code for model monitoring on local device
model_execution_monitoring_prompt = {
    "messages": [
        {
            "role": "user",
            "content": "Third, based on the Python script you wrote, please add a monitoring function for the script to monitor the maximum CPU memory usage, the maximum GPU memory usage, and the size of the model used for local storage. Please directly output the code without extra contents. Below are some reference code snippets for system resource related monitoring:\n\n<reference_code>\n{}\n</reference_code>",
        }
    ]
}

#### Task 1.3.3: modify the code by loading model parameters from local disk
model_inference_script_generation_prompt = {
    "messages": [
        {
            "role": "user",
            "content": "Finally, please modify the code you generated, where the model is directly loaded from a local file named '{}.pth'. The loaded model can be directly used for inference. Please directly output the modified code without additional contents.",
        }
    ]
}

### Task 1.4: Filter the program output and only keep the useful information
program_output_filtering_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a professional Python programmer. Given the program output, please filter out irrelevant information (e.g., warning, error, etc.). The useful information only includes the resource or performance monitoring results. Please directly output the extracted information.",
        ),
        (
            "human",
            "Please filter and extract the useful information from the following program output:\n{input}",
        ),
    ]
)

# %% Task 2: Search space determination based on model profiling results
### Task 2.1: Generate different optimization methods
#### Task 2.1.1: Generate a list of potential hyperparameters that can be optimized
optimization_method_generation_prompt = {
    "messages": [
        {
            "role": "system",
            "content": "You are a professional Python programmer, especially in the field of machine learning. The user wants to deploy a well-trained machine learning model on an edge device with limited resources. First, I need to deploy the model on my local PC and monitor the resource consumption of the original model during inference. Next, I need to optimize the model to make sure that the edge device can afford storing and executing the optimized model. Therefore, you are tasked with optimizing the original model for deploying it on resource-constrained edge devices.\n\nLet's think step by step. First, based on the deployment code of the original model and the resource monitoring results provided by the user, please first analyze the model deployment code and then generate a list of hyperparameters that can be optimized. You should adopt the following format for your response without extra content:\nHyperparameter name 1\nHyperparameter name 2\n......",
        },
        {
            "role": "user",
            "content": "<model_deployment_code>\n```python\n{}\n```\n\n<resource_monitoring_results>\n{}\n</resource_monitoring_results>",
        },
    ]
}

#### Task 2.1.2: Filter out those hyperparameters that are not suitable for optimization
#### For example, modifying model parameters and architecture is not allowed
hyperparameter_filtering_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            'You are a professional Python programmer, especially in the field of machine leaning. Given the user input of an optimizable hyperparameter for a well-trained machine learning model, please judge whether the hyperparameter meet the following conditions:\n1. The hyperparameter should be related to resource consumption (i.e., GPU memory usage, CPU memory usage, model size, throughput) during model inference.\n2. The machine learning model is already well-trained and does not need any re-training processes. Therefore, only hyperparameters that can be modified during inference should be considered.\n3. No new models should be created, i.e., the optimization method should be directly applied on the original model instead of creating new one.\n4. Any hyperparameters related to input data must not be considered, since the dimension of the input and output of the model is fixed. For example, reducing the size/resolution of input data should not be considered.\n5. The model will be deployed on {device}. Therefore, please make sure that the optimization hyperparameter is applicable when running on {device}.\n\nPlease only output "Yes" if the hyperparameter meets the requirements; otherwise, output "No". Directly output the result without extra contents.',
        ),
        (
            "human",
            "<optimizable_hyperparameter>{input}</optimizable_hyperparameter>",
        ),
    ]
)

#### Task 2.1.3: Make sure that each hyperparameter can be optimized by the Optuna package
optimization_method_validation_prompt = {
    "messages": [
        {
            "role": "system",
            "content": "You are a professional Python programmer, especially in the field of machine leaning. Given the user input of an optimizable hyperparameter for a well-trained machine learning model, please judge whether the hyperparameter can be tuned via the Optuna package without additional training. You must use the web search tool to ensure that the hyperparameter can be optimized by the Optuna package. Please only output 'Yes' if the hyperparameter can be optimized by the Optuna package; otherwise, output 'No'. Directly output the result without extra contents.",
        },
        {
            "role": "user",
            "content": "<optimizable_hyperparameter>{}</optimizable_hyperparameter>",
        },
    ]
}

#### Task 2.1.3: Analyze the hyperparameters of the original model
hyperparameter_analysis_prompt = {
    "messages": [
        {
            "role": "system",
            "content": "You are a professional Python programmer, especially in the field of machine learning and model optimization. Based on the following information, please answer the user's question and you can use the web search tool.\n\n1. A Python script that can deploy a model and perform inference on a local dataset.\n<model_deployment_code>\n```python\n{}\n```\n</model_deployment_code>\n2. A list of hyperparameters:\n<hyperparameter_list>\n{}\n</hyperparameter_list>",
        },
        {
            "role": "user",
            "content": "Based on the provided information, please first analyze the model deployment code and extract the value of each hyperparameter in the hyperparameter list. Then, please directly output the value of these hyperparameters without additional contents. Please refer to the following output format:\n* Hyperparameter 1: Value 1 or None (if not applicable)\n* Hyperparameter 2: Value 2 or None (if not applicable\n......",
        },
    ]
}

### Task 2.2: Generate search space for each optimization method
#### Task 2.2.1: Apply a random value of optimization hyperparameter to the model
model_optimization_adoption_prompt = {
    "messages": [
        {
            "role": "system",
            "content": "You are a professional Python programmer, especially in the field of machine learning. Below is an sample code for deploying a machine learning model and performing inference on a local dataset. Based on the deployment code, please answer the user's question. You should use the web search tool if necessary. Additionally, for model pruning or quantization related techniques, you must retrieve relevant information and sample code from the database.\n\n<model_deployment_code>\n```python\n{}\n```",
        },
        {
            "role": "user",
            "content": "Based on the provided information, you are tasked with generating a Python script that finds the optimal {} value to minimize the resource consumption when deploying the model and performing inference tasks. Let's think step by step. First, based on the provided model deployment code, please select a {} and apply it to the model. Please note that the model will be executed on {}. Please directly output the code without additional contents.",
        },
    ]
}

#### Task 2.2.2: Generate search space for each hyperparameter
search_space_generation_prompt = {
    "messages": [
        {
            "role": "user",
            "content": "Second, please modify the code you generated by creating a search space for different values of {} using the Optuna package. Note that for model pruning or quantization related techniques, you must retrieve relevant information and sample code from the database. Please directly output the modified code without additional contents",
        }
    ]
}

#### Task 2.2.3: Modify the objective function based on user requirements
objective_function_modification_prompt = {
    "messages": [
        {
            "role": "user",
            "content": "Third, please modify the objective function of the code you generated based on the following information:\n1. The user has multiple resource consumption requirements for deploying the model:\n{}\n2. You must change the objective function to multi-objective objective function, where each resource consumption requirement is considered as an optimization objective.\n3. Here are some sample code snippets for resource monitoring and you must incorporate all the code:\n{}\n\nPlease directly output the modified code without additional contents",
        }
    ]
}

#### Task 2.2.4: Modify the search scope and print out the optimal hyperparameter
search_scope_modification_prompt = {
    "messages": [
        {
            "role": "user",
            "content": "Then, please modify the search scope (i.e., suggested values in Optuna) of {} based on the following information:\n1. The user's resource requirement for deploying the model:\n{}\n2. The default hyperparameter configurations of the original model deployment code:\n{}\n3. The resource monitoring results when deploying the original model:\n{}\n\nPlease comprehensively consider the search scope of the hyperparameter. Please directly output the modified code without additional contents.",
        }
    ]
}

#### Task 2.2.5: Modify the code that output all possible hyperparameters
output_all_possible_hyperparameters_prompt = {
    "messages": [
        {
            "role": "user",
            "content": "Finally, please modify the code you generated based on the following requirements:\n1. Go through all the trials and judge whether all the resource requirements are satisfied within that trial.\n2. If all resource requirements are satisfied, the code should output the hyperparameter configuration of the trial. Please directly output the final code without additional contents.",
        }
    ]
}

# %% Task 3: Model optimization code generation by applying various hyperparameter configurations
### Task 3.1 Generate optimization code by combining all spaces and optimization methods
search_scope_combination_prompt = {
    "messages": [
        {
            "role": "system",
            "content": "You are a professional Python programmer, especially in the field of machine learning and model optimization. Based on the Python script provided below, please answer the user's question and you should use the web search tool if necessary.\n\nA Python script for deploying a machine learning model and performing inference on a local dataset:\n<model_deployment_code>\n```python\n{}\n```\n</model_deployment_code>",
        },
        {
            "role": "user",
            "content": "Based on the provided code, please write a Python script that creates an objective function to tune the {}. Please directly output the code without additional contents. You must refer to the following code snippet for reference:\n\n<code_for_{}_tuning>\n```python\n{}\n```\n</code_for_{}_tuning>",
        },
    ]
}

search_scope_addition_prompt = {
    "messages": [
        {
            "role": "user",
            "content": "Next, please continue to modify the code you generated by adding one more hyperparameter according to the following code. Please directly output the modified code without additional contents.\n\n<code_for_{}_tuning>\n```python\n{}\n```\n</code_for_{}_tuning>",
        }
    ]
}

advanced_search_strategy_prompt = {
    "messages": [
        {
            "role": "user",
            "content": "Finally, please modify the code you generated by adopting:\n1. A larger trial number.\n2. A more advanced sampler for hyperparameter searching.\n3. A more advanced pruner. Please directly output the modified code without additional contents.",
        }
    ]
}

### Task 3.3: Generate corresponding local deployment code for each configuration
local_optimized_model_deployment_code_generation_prompt = [
    {
        "role": "system",
        "content": "You are a professional Python programmer, especially in the field of machine learning. Based on the model deployment code of the {} model and the configurations for model optimization, please write a Python script that deploys the model with the optimized hyperparameters. Please directly output the code without extra contents.",
    },
    {
        "role": "user",
        "content": "Code for deploying the original model:\n<original_model_deployment>\n```python\n{}\n```\n\n</original_model_deployment>\nModel optimization configurations:\n<optimization_configurations>\n{}\n</optimization_configurations>",
    },
]


# %% Task 4: Model deployment on edge device
### Task 4.1: Judge whether the optimization hyperparameter is applied to model or not
hyperparameter_judgement_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a professional Python programmer, especially in the field of machine learning and model optimization. Given the user input of an optimization hyperparameter for a well-trained machine learning model, please judge whether the hyperparameter can be applied to the model or to other objects (e.g, dataset). If the hyperparameter can be applied to the model, output 'Yes'; otherwise, output 'No'. Directly output the result without extra contents.",
        ),
        (
            "human",
            "<optimization_hyperparameter>{input}</optimization_hyperparameter>",
        ),
    ]
)

### Task 4.2: Apply model optimization for each hyperparameter configuration
model_deployment_code_rewrite_prompt = {
    "messages": [
        {
            "role": "system",
            "content": "You are a professional Python programmer, especially in the field of machine learning and mode optimization. Based on the Python script below  Based on the following information, please answer the user's question. You can use the web search tool if necessary.\n\n1. A Python script that can deploy a machine learning model and perform inference on a local dataset.\n<model_deployment_code>\n```python\n{}\n```\n</model_deployment_code> \n\n2. A Python script that can optimize the model by adopting various hyperparameters.\n<local_optimization_code>\n```python\n{}\n```\n</local_optimization_code>",
        },
        {
            "role": "user",
            "content": "Based on the provided model deployment code, please generate a new Python script that loads the same model as the sample code and save the entire model to a local file named 'model_{}.pth'. Please directly output the code without additional contents.",
        },
    ]
}

model_optimization_addition_prompt = {
    "messages": [
        {
            "role": "user",
            "content": "Next, please modify the code you generated by applying a hyperparameter to the model for optimization. The hyperparameter is {} with its value as {}. If the value of {} is 'False', please do not modify code. If the hyperparameter is related to pruning, please  You must refer the provided local optimization code when adopting the hyperparameter. Please directly output the code without additional contents.",
        }
    ]
}

# %% Task 5: Transform the model to the target platform
### Task 5.1: first write some code to transform the PyTorch into ONNX representation
model_transformation_prompt = {
    "messages": [
        {
            "role": "system",
            "content": "You are a professional Python programmer, especially in the field of machine learning and model deployment on edge devices. The user wants to deploy a machine learning model on an edge device to perform inference tasks. Based on the provided information below, please answer the user's question. You should use the web search tool if necessary.\n\nThe dataset information of the model:\n<dataset_information>\n{}\n</dataset_information>",
        },
        {
            "role": "user",
            "content": "Please first generate a Python script that loads a machine learning model from a local file named 'model_{}.pth'. Then, please export the model as ONNX representation to a local file named 'model_{}.onnx'. Note that the model should be executed on {}. Please directly output the code without additional contents.",
        },
    ]
}

### Task 5.2: Apply non-model optimization method
non_model_optimization_prompt = {
    "messages": [
        {
            "role": "user",
            "content": "Next, based on the code you generated, please modify the code by applying the following hyperparameters with their values to the code. Please directly output the code without additional contents.\n\nA list of hyperparameters:\n<hyperparameter_list>\n{}\n</hyperparameter_list>",
        }
    ]
}


# %% Task 6: Generate code for performance monitoring
### Task 6.2: generate code for local deployment and monitoring
#### Task 6.2.1: generate code for onnx model loading
ONNX_model_loading_prompt = {
    "messages": [
        {
            "role": "system",
            "content": "You are a professional Python programmer, especially in the field of machine learning and model deployment on edge devices. Please answer the user's question and you should use the web search tool if necessary.",
        },
        {
            "role": "user",
            "content": "Please write a Python script that loads an ONNX model from a local file named 'model_file.onnx'. Please directly output the code without additional contents.",
        },
    ]
}

#### Task 6.2.2: generate code for model inference on local dataset
edge_model_inference_prompt = {
    "messages": [
        {
            "role": "user",
            "content": "Next, based on the code you generated, please modify the code by performing model inference on a local dataset based on the following information. Please only load the testing dataset. Please directly output the modified code without additional contents.\n\n1. The information of the local dataset for inference.\n<dataset_information>\n{}\n</dataset_information>\n\n2. The path to the dataset:\n<dataset_path>{}</dataset_path>\nThe test data is stored in a numpy array named 'test_data.npy' and the test labels are stored in a numpy array named 'test_label.npy'.\n\n3. A list of hyperparameters that should be applied to the input data of the model.\n<hyperparameters>\n{}\n</hyperparameters>",
        }
    ]
}
#### Task 6.2.3: add two arguments: model weight path and dataset path
edge_model_argument_addition_prompt = {
    "messages": [
        {
            "role": "user",
            "content": "Next, based on the code you generated, please modify it by converting the model weight file path and the dataset path into two command-line arguments. The name of these two arguments should be 'model_weight_path' and 'dataset_path'. Please directly output the modified code without additional contents.",
        }
    ]
}

#### Task 6.2.4: make the non model optimization hyperparameter as input arugments
non_model_optimization_argument_prompt = {
    "messages": [
        {
            "role": "user",
            "content": "Then, based on the code you generated, please modify it by converting each of the following hyperparameters into command-line arguments. The name of the arguments should be the same as the hyperparameter name. Please directly output the modified code without additional contents.\n\n<hyperparameter_list>\n{}\n</hyperparameter_list>",
        }
    ]
}

#### Task 6.2.5: add resource monitoring code
edge_resource_monitoring_prompt = {
    "messages": [
        {
            "role": "user",
            "content": "Then, based on the code you generated, please modify the code by adding a function to monitor the resource consumption of the model during inference. Please directly output the modified code without additional contents. Below is the the reference code for resource monitoring:\n<reference_code>\n{}\n</reference_code>",
        }
    ]
}

#### Task 6.2.6: add performance monitoring code
edge_performance_monitoring_prompt = {
    "messages": [
        {
            "role": "user",
            "content": "Next, based on the code you generated, please modify the code by adding a function to monitor the model's performance during inference. Please directly output the modified code without additional contents. Below is the reference code for model performance monitoring:\n<reference_code>\n{}\n</reference_code>",
        }
    ]
}

# %% Task 7: Cross platform deployment and monitoring
### Task 7.4: model execution result filtering
model_monitoring_output_filtering_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a professional Python programmer, especially in the field of machine learning and model deployment on edge devices. The user will provide the performance result of a program that deploys a machine learning model on an edge device and performs inference on a local dataset. You are tasked with filtering out irrelevant information (e.g., warning, error, etc.). The useful information only includes the performance monitoring results of the model. Please directly output the extracted information without additional contents.",
        ),
        (
            "human",
            "Please filter and extract the useful information from the following program output:\n{input}",
        ),
    ]
)

### Task 7.5: Judge whether the model performance meets the requirements
### Task 7.4: model execution result filtering
performance_requirement_judgement_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a professional Python programmer, especially in the field of machine learning and model deployment on edge devices. The user deployed a machine learning model on an edge device and monitoring the performance of the model when perform inference on a local dataset. Please answer the user's question based on the following information:\n1. The user's requirements for model performance:\n{performance_requirement}\n\n2. The performance monitoring result of the model when performing inference on the local dataset:\n{performance_monitoring_result}",
        ),
        (
            "human",
            "Based on the provided information, please judge whether the model meets all of the user's requirements. If the model meets all the requirements, output 'Yes'; otherwise, output 'No'. Directly output the result without additional contents.",
        ),
    ]
)
