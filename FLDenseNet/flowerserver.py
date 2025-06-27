import flwr as fl
import numpy as np
import os
import pandas as pd

# from tensorflow import keras
# from keras import layers
# from keras.applications.densenet import DenseNet121
from typing import List, Tuple, Dict, Optional
from flwr.common import Parameters, Scalar, Metrics, ndarrays_to_parameters

# DATA_SPLIT = '50-50'
# DATA_SPLIT = '70-30'
# DATA_SPLIT = '90-10'
DATA_SPLIT = 'DIFF'

IMG_SIZE = 256
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
PATH_TO_BEST_ACC = './accuracy/best-metrics.npz'

best_accuracy = np.load(PATH_TO_BEST_ACC, allow_pickle=True)['arr_0'] if os.path.exists(PATH_TO_BEST_ACC) else 0

# USE THIS TO RESUME LAST SESSION
# PATH_TO_BEST_PARAMS = './weights/round-1-weights.npz'
# loaded_best_parameters = np.load(PATH_TO_BEST_PARAMS) if os.path.exists(PATH_TO_BEST_PARAMS) else None
# if loaded_best_parameters is not None:
#     for f in loaded_best_parameters.files:
#         best_parameters.append(loaded_best_parameters[f]) 
best_parameters=[]

# USED TO SAVE MODEL -- KALO SERVER ADA PADA SATU MESIN YANG BERBEDA
# def ClsModel(n_classes=1):
#     base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
#     x = layers.AveragePooling2D(pool_size=(3,3), name='avg_pool')(base_model.output)
#     x = layers.Flatten()(x)
#     x = layers.Dense(128, activation='relu', name='dense_post_pool')(x)
#     x = layers.Dropout(0.5)(x)
#     outputs = layers.Dense(n_classes, activation='sigmoid', name='predictions')(x)
#     model = keras.Model(inputs=base_model.input, outputs = outputs)
    
#     return model

# model = ClsModel()

history = dict()
history['accuracy'] = []
history['loss'] = []
history['val_accuracy'] = []
history['val_loss'] = []
history['test_f1_score'] = []

class SaveModelAndMetricsStrategy(fl.server.strategy.FedAvg):
    def aggregate_evaluate(
            self,
            rnd: int, 
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
            failures: List[BaseException],
        ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        aggregated_tuple = super().aggregate_evaluate(rnd, results, failures)
        aggregated_loss, aggregated_metrics = aggregated_tuple

        if aggregated_metrics is not None and aggregated_loss is not None:
            history['test_f1_score'].append(aggregated_metrics["test_f1_score"])

        return aggregated_tuple
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]], # FitRes is like EvaluateRes and has a metrics key 
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        """Aggregate model weights using weighted average and store checkpoint"""
        aggregated_tuple = super().aggregate_fit(rnd, results, failures)
        aggregated_parameters, aggregated_metrics = aggregated_tuple

        if aggregated_metrics is not None and aggregated_parameters is not None:
            history['accuracy'].append(aggregated_metrics["accuracy"])
            history['loss'].append(aggregated_metrics["loss"])
            history['val_accuracy'].append(aggregated_metrics["val_accuracy"])
            history['val_loss'].append(aggregated_metrics["val_loss"])
       
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Save aggregated_ndarrays
            print(f"Saving {rnd} round aggregated_ndarrays...")
            np.savez("./weights/" + DATA_SPLIT + f"/round-{rnd}-weights.npz", *aggregated_ndarrays)

        return aggregated_tuple 
    
def fit_weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    val_accuracies = [num_examples * m["val_accuracy"] for num_examples, m in metrics]
    val_losses = [num_examples * m["val_loss"] for num_examples, m in metrics]

    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples), "loss": sum(losses) / sum(examples), "val_accuracy": sum(val_accuracies) / sum(examples), "val_loss": sum(val_losses) / sum(examples)}

def eval_weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    f1_scores = [num_examples * m["f1_score"] for num_examples, m in metrics]

    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"test_f1_score": sum(f1_scores) / sum(examples)}

if best_parameters:
    strategy = SaveModelAndMetricsStrategy(
        min_available_clients=2,  # Minimum number of clients that need to be connected to the server before a training round can start
        initial_parameters=ndarrays_to_parameters(best_parameters),
        fit_metrics_aggregation_fn=fit_weighted_average,
        evaluate_metrics_aggregation_fn=eval_weighted_average
    )
else:
    strategy = SaveModelAndMetricsStrategy(
        min_available_clients=2,  # Minimum number of clients that need to be connected to the server before a training round can start
        fit_metrics_aggregation_fn=fit_weighted_average,
        evaluate_metrics_aggregation_fn=eval_weighted_average
    )

fl.server.start_server(
    server_address="10.30.200.41:5002",
    config=fl.server.ServerConfig(num_rounds=9),
    grpc_max_message_length=1024*1024*1024,
    strategy=strategy
)

df = pd.DataFrame(history)
df.to_csv(f"./metrics/history-{DATA_SPLIT}.csv")
df