# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Install Flower
RUN pip install flwr

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the server script into the container at /usr/src/app
COPY server.py .

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Run server.py when the container launches
CMD ["python", "server.py"]
