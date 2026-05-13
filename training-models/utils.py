"""
utils.py: training, evaluation, visualization, and testing utilities
shared across all dermamnist versions.
"""

import os

import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.metrics import classification_report
from medmnist import INFO
from torchinfo import summary as torchinfo_summary
from torchview import draw_graph


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader) -> dict:
    """
    Evaluate model on loader; prints a classification report and returns
    a dict with 'accuracy'.
    """
    data_flag = "dermamnist"
    info = INFO[data_flag]

    model.eval()
    correct = 0
    total = 0
    label_list = []
    pred_list = []

    for data in loader:
        images, labels = data[0], data[1]
        labels = labels.squeeze().long()

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        pred_list.extend(predicted.numpy().tolist())
        label_list.extend(labels.numpy().tolist())

    print(
        classification_report(
            label_list, pred_list, target_names=list(info["label"].values()), digits=4
        )
    )
    return {"accuracy": correct / total}


def train(
    model: nn.Module,
    optimizer,
    loader_train: DataLoader,
    loader_val: DataLoader,
    output_path: str,
    num_epochs: int = 100,
    max_patience: int = None,
    use_tensorboard: bool = False,
):
    """
    Train model, saving the best checkpoint to output_path/best_model.pt.

    max_patience: if set, training stops early when val accuracy stops improving.
    use_tensorboard: if True logs to TensorBoard; otherwise saves matplotlib plots.
    """
    os.makedirs(output_path, exist_ok=True)

    loss_function = torch.nn.CrossEntropyLoss()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")

    iteration = 0
    epoch_length = len(loader_train)
    patience = max_patience
    best_accuracy = 0

    if use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        summary = SummaryWriter(output_path, purge_step=0)
        summary.add_scalar("num_params", num_params)
    else:
        train_loss = []
        val_acc = []

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1}/{num_epochs}...")

        for batch in tqdm(loader_train):
            iteration += 1

            model.train()
            inputs, labels = batch[0], batch[1].squeeze().long()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            if (iteration % 50) == 0:
                if use_tensorboard:
                    summary.add_scalar("loss/train", loss, iteration)
                else:
                    train_loss.append((loss.detach().item(), iteration))

            if (iteration % epoch_length) == 0:
                if use_tensorboard:
                    train_metrics = evaluate_model(model, loader_train)
                    for key, val in train_metrics.items():
                        summary.add_scalar(f"{key}/train", val, iteration)

                    val_batch = next(iter(loader_val))
                    val_inputs = val_batch[0]
                    val_labels = val_batch[1].squeeze().long()
                    optimizer.zero_grad()
                    val_outputs = model(val_inputs)
                    val_loss = loss_function(val_outputs, val_labels)
                    summary.add_scalar("loss/val", val_loss, iteration)

                val_metrics = evaluate_model(model, loader_val)
                accuracy = val_metrics["accuracy"]

                if use_tensorboard:
                    for key, val in val_metrics.items():
                        summary.add_scalar(f"{key}/val", val, iteration)
                else:
                    val_acc.append((accuracy, iteration))

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    if max_patience is not None:
                        patience = max_patience
                    torch.save(model.state_dict(), os.path.join(output_path, "best_model.pt"))
                elif max_patience is not None:
                    patience -= 1

                if max_patience is not None:
                    print(f"My remaining patience is {patience}.")
                print(f"Current accuracy is {accuracy}, and best is: {best_accuracy}.")

                if max_patience is not None and patience == 0:
                    print("My validation patience ran out.")
                    return

    if not use_tensorboard:
        plt.figure()
        plt.plot([it for _, it in train_loss], [loss for loss, _ in train_loss])
        plt.xlabel("Iterations")
        plt.ylabel("Training loss")
        plt.grid()
        plt.savefig(os.path.join(output_path, "train_loss.png"))

        plt.figure()
        plt.plot([it for _, it in val_acc], [acc for acc, _ in val_acc])
        plt.xlabel("Iterations")
        plt.ylabel("Validation accuracy")
        plt.grid()
        plt.savefig(os.path.join(output_path, "val_acc.png"))


def visualize_model(model: nn.Module, output_path: str):
    """
    Save model summary (torchinfo), computation graph (torchview), and ONNX
    export (for netron) to output_path.
    """
    os.makedirs(output_path, exist_ok=True)

    model_stats = torchinfo_summary(model, input_size=(1, 3, 32, 32), verbose=0)
    with open(os.path.join(output_path, "model_summary.txt"), "w") as f:
        f.write(str(model_stats))
    print(f"Model summary saved to {output_path}/model_summary.txt")

    graph = draw_graph(
        model,
        input_size=(1, 3, 32, 32),
        save_graph=True,
        filename="model_graph",
        directory=output_path,
        expand_nested=True,
    )
    graph.visual_graph.render(format="png", cleanup=True)
    print(f"Model graph saved to {output_path}/model_graph.png")

    dummy_input = torch.randn(1, 3, 32, 32)
    onnx_path = os.path.join(output_path, "model.onnx")
    torch.onnx.export(model, dummy_input, onnx_path, opset_version=11)
    print(f"ONNX model saved. Run: netron {onnx_path}")


def test(model_class, output_path: str, batch_size: int = 8):
    """
    Instantiate model_class, load the best checkpoint from output_path, and
    evaluate on the test set.
    """
    from data import load_datasets

    model = model_class()
    model.load_state_dict(torch.load(os.path.join(output_path, "best_model.pt")))

    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")

    _, data_test = load_datasets("test")
    loader_test = DataLoader(data_test, batch_size=batch_size, shuffle=False)

    metrics = evaluate_model(model, loader_test)
    print(f"Test accuracy is: {metrics['accuracy']}")
