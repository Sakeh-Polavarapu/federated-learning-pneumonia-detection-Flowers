Federated Learning with CIFAR-100 using Flower and PyTorch


This project implements a Federated Learning (FL) system using the CIFAR-100 dataset, Flower framework, and PyTorch. The setup simulates three clients collaboratively training a ResNet34 model in a privacy-preserving, decentralized manner. Each client is containerized using Docker and holds a disjoint but class-complete subset of CIFAR-100 data.

Key Features
Simulates 3 clients with class-balanced CIFAR-100 data

Uses Flower for federated coordination and communication

ResNet34 architecture modified to use GroupNorm (BatchNorm disabled)

Docker-based setup for isolated and reproducible experimentation

Central server performs Federated Averaging (FedAvg) aggregation


Dataset
CIFAR-100: A dataset of 60,000 32x32 color images across 100 classes.
In this project, it is split across 3 clients such that each client receives a different subset, but all 100 classes are included per client.
