import os
from flask import Flask, request

local_dataset_path = "./dataset/ILSVRC"

app = Flask(__name__)


# transmit the model weight to the edge device
@app.route("/model_transmission", methods=["POST"])
def model_transmission():
    try:
        meta_information = request.form
        print('Received information:')
        print(request.form)

        model_name = meta_information["model_name"]
        model_weight_name = meta_information["model_weight_name"]
        model_size = meta_information["model_size"]

        model_weight_file = request.files["file"]
        model_weight_file.save(model_weight_name)

        print(
            "Received model {}'s weight file: {} | size: {}".format(
                model_name, model_weight_name, model_size
            )
        )

        return {"status": True, "message": "Model weight received successfully."}
    except Exception as e:
        return {"status": False, "message": str(e)}


# transmit the script to the edge device
@app.route("/script_transmission", methods=["POST"])
def script_transmission():
    try:
        meta_information = request.form
        print('Received information:')
        print(request.form)

        script_name = meta_information["script_name"]
        script_file = request.files["file"]
        script_file.save("./{}".format(script_name))

        print("Received script file: {}".format(script_name))

        return {"status": True, "message": "Script received successfully."}
    except Exception as e:
        return {"status": False, "message": str(e)}


# execute the script to deploy the model on the edge device
@app.route("/model_execution", methods=["POST"])
def model_execution():
    try:
        meta_information = request.form
        print('Received information:')
        print(request.form)
        
        command = meta_information["command"]
        model_weight_name = meta_information["model_weight_name"]
        script_name = meta_information['script_name']
        command = command.format(script_name, model_weight_name, local_dataset_path)
        os.system(command)

        execution_result = open("log.txt", "r", encoding="utf-8").read()

        # after finishing model inference task, delete model and script
        os.remove(model_weight_name)
        os.remove(script_name)

        return {
            "status": True,
            "message": "Model execution completed.",
            "result": execution_result,
        }
    except Exception as e:
        return {"status": False, "message": str(e), "result": "NULL"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
