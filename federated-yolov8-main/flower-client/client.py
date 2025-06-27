import flwr as fl
from ultralytics import YOLO
import torch
from collections import OrderedDict
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class YoloV8Client(fl.client.NumPyClient):
    def __init__(self):
        logger.info("Initializing YOLOv8 client")
        self.model = YOLO('yolov8n.yaml')
        self.model = YOLO('yolov8n.pt')

    def get_parameters(self, config):
        logger.info("Getting model parameters")
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        logger.info("Setting model parameters")
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        logger.info("Starting training")
        self.set_parameters(parameters)
        results = self.model.train(data='coco128.yaml', epochs=3)
        logger.info(f"Training results: {results}")
        return self.get_parameters(), results.size, {}

    def evaluate(self, parameters, config):
        logger.info("Starting evaluation")
        self.set_parameters(parameters)
        results = self.model.val()
        logger.info(f"Evaluation results: {results}")

        # Example evaluation metric calculation
        # Modify this according to your needs
        accuracy = results.metrics.get('accuracy', 0.0)
        return float(accuracy), results.size, {}

# Initialize and start the YOLOv8 Flower client
client = YoloV8Client()
fl.client.start_numpy_client(server_address="flower_server:8080", client=client)
