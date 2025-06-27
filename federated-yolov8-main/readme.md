# DOCUMENTATION
sudo docker compose up --build

## Libraries
- Yolov8
- Flower (Federated Learning Framework)
- Docker for deployment

## Project Structure
federated-yolov8/
│
├── docker-compose.yml
├── flower-client/
│   ├── Dockerfile
│   └── client.py
├── flower-server/
│   ├── Dockerfile
│   └── server.py
├── nodes/
│   ├── data_node_1/
│   │   ├── data.yaml
│   │   ├── test/
│   │   ├── train/
│   │   └── valid/
│   ├── data_node_2/
│   │   ├── data.yaml
│   │   ├── test/
│   │   ├── train/
│   │   └── valid/
│   └── data_node_3/
│       ├── data.yaml
│       ├── test/
│       ├── train/
│       └── valid/
└── readme.md