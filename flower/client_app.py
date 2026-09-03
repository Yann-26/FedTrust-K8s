import torch

from flwr.client import ClientApp
from flwr.common import (
    ArrayRecord,
    Context,
    Message,
    MetricRecord,
    RecordDict,
)

from .task import (
    SimpleCNN,
    get_client_loaders,
    train_model,
)


# ============================================================
# Flower Client Application
# ============================================================

app = ClientApp()


@app.train()
def train(
    msg: Message,
    context: Context,
):
    """Train the global model on one client's local MNIST data."""

    # --------------------------------------------------------
    # Client identity
    # --------------------------------------------------------

    partition_id = int(context.node_config["partition-id"])

    num_partitions = int(context.node_config["num-partitions"])

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    batch_size = int(context.run_config["batch-size"])

    local_epochs = int(context.run_config["local-epochs"])

    learning_rate = float(context.run_config["learning-rate"])

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = SimpleCNN()

    state_dict = msg.content["arrays"].to_torch_state_dict()

    model.load_state_dict(state_dict)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    # --------------------------------------------------------
    # Local dataset
    # --------------------------------------------------------

    trainloader = get_client_loaders(
        client_id=partition_id,
        num_clients=num_partitions,
        batch_size=batch_size,
    )

    # --------------------------------------------------------
    # Local training
    # --------------------------------------------------------

    train_loss = train_model(
        model=model,
        trainloader=trainloader,
        epochs=local_epochs,
        learning_rate=learning_rate,
    )

    # --------------------------------------------------------
    # Return updated model
    # --------------------------------------------------------

    arrays = ArrayRecord(model.state_dict())

    metrics = MetricRecord(
        {
            "train_loss": float(train_loss),
            "num-examples": len(trainloader.dataset),
        }
    )

    content = RecordDict(
        {
            "arrays": arrays,
            "metrics": metrics,
        }
    )

    return Message(
        content=content,
        reply_to=msg,
    )
