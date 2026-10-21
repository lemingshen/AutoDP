from transformers import ResNetForImageClassification
import torch
import torch_pruning as tp

def save_optimized_model(pruning_ratio, quantization):
    # Load the model
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-152")
    model.eval()

    if pruning_ratio:
        # Example input for the model
        example_inputs = torch.randn(1, 3, 224, 224)

        # Set up the importance criterion for pruning
        imp = tp.importance.GroupNormImportance(p=2)

        # Identify layers to ignore during pruning
        ignored_layers = []
        for m in model.modules():
            if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
                ignored_layers.append(m)

        # Initialize the pruner
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

    # Save the entire model to a local file
    torch.save(model, 'model_95.pth')

if __name__ == "__main__":
    # Set the hyperparameter values
    pruning_ratio = 0.3892679112465522
    quantization = False
    save_optimized_model(pruning_ratio, quantization)