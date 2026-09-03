import torch

from flwr.common import (
    ArrayRecord,
    ConfigRecord,
)

from flwr.server import ServerApp
from flwr.serverapp import Grid
from flwr.server.strategy import FedAvg

from .task import (
    SimpleCNN,
    get_testloader,
    evaluate_model,
)


app = ServerApp()


def evaluate_global_model(
    arrays,
):
    """Evaluate the global model on the complete MNIST test set."""

    model = SimpleCNN()

    state_dict = arrays.to_torch_state_dict()

    model.load_state_dict(state_dict)

    testloader = get_testloader()

    loss, accuracy = evaluate_model(
        model,
        testloader,
    )

    return {
        "test_loss": float(loss),
        "test_accuracy": float(accuracy),
    }


@app.main()
def main(
    grid: Grid,
    context,
):
    """Start the Flower federated learning server."""

    num_rounds = int(context.run_config["num-server-rounds"])

    learning_rate = float(context.run_config["learning-rate"])

    global_model = SimpleCNN()

    arrays = ArrayRecord(global_model.state_dict())

    strategy = FedAvg(
        fraction_train=1.0,
        fraction_evaluate=0.0,
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord(
            {
                "learning-rate": learning_rate,
            }
        ),
        num_rounds=num_rounds,
        evaluate_fn=evaluate_global_model,
    )

    final_state_dict = result.arrays.to_torch_state_dict()

    torch.save(
        final_state_dict,
        "results/flower_exp10_final_model.pt",
    )

    print("\nFlower Experiment 10 completed.")
