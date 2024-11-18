
# Federated Learning with MNIST

This project demonstrates a simple Federated Learning framework using the MNIST dataset and PyTorch. 

## Features
- Simulates federated learning with 5 clients.
- Uses a simple neural network to classify MNIST digits.
- Implements the Federated Averaging (FedAvg) algorithm.

## Setup
1. Install the required libraries:
   ```
   pip install torch torchvision
   ```
2. Run the script:
   ```
   python main.py
   ```

## Results
The global model achieves ~97.91% accuracy on the MNIST test dataset after 5 federated learning rounds.


## Evaluation Metrics
The accuracy of the global model improved over 5 rounds of federated learning, as shown below:

![Evaluation Metrics](evaluation_metrics_slide.png)
