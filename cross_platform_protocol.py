import requests


def send_file(file_path, url, meta_information):
    """
    Case 1: model transmission
    url - "http://localhost:8080/model_transmission"
    meta_information = {
        "model_name": "ResNet152",
        "model_weight_name": "resnet152_weights.pth",
        "model_size": "152MB",
    }
    Case 2: script transmission
    url - "http://localhost:8080/script_transmission"
    meta_information = {"script_name": "script.py"}
    """
    with open(file_path, "rb") as file:
        response = requests.post(url, data=meta_information, files={"file": file})
        print(response)
        if response.status_code == 200 and response.json()["status"]:
            returned_data = response.json()
            print(
                "Status: {} | Message: {}".format(
                    returned_data["status"], returned_data["message"]
                )
            )
        else:
            print("Failed to send file.")


def send_command(url, meta_information):
    """
    url - "http://localhost:8080/model_execution"
    command = "python3 script.py --model_weight_path {} --dataset_path {} > log.txt"
    """
    response = requests.post(url, data=meta_information)
    print(response)
    if response.status_code == 200 and response.json()["status"]:
        returned_data = response.json()
        print(
            "Status: {} | Message: {} | Result: {}".format(
                returned_data["status"],
                returned_data["message"],
                returned_data["result"],
            )
        )

        return returned_data["result"]
    else:
        print("Failed to send command.")
