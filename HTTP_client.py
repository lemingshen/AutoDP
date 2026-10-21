import cross_platform_protocol

execution_result = cross_platform_protocol.send_command(
    url="{}/{}/model_execution".format(""),
    meta_information={
        "command": "python3 edge_deployment_{}.py --model_weight_path {} --dataset_path {}".format(
            2, "{}", "{}"
        ),
        "model_weight_name": "model_{}.onnx".format(index),
    },
)
