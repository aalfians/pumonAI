# Use the Ultralytics base image
FROM ultralytics/ultralytics:8.0.229

# Install Flower
RUN pip install flwr

# Install GPU
# https://pytorch.org/get-started/locally/
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy the client script and the wait-for-it script
COPY client.py .
COPY wait-for-it.sh .
RUN chmod +x wait-for-it.sh

CMD ["./wait-for-it.sh", "flower_server:8080", "--", "python", "client.py"]
