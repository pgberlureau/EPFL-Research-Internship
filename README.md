# Long-range Propagation in Graph Neural Networks with Learnable Geometry and Unitary Operators

This repository contains code from my research internship on improving long-range information propagation in Graph Neural Networks (GNNs). We propose two novel architectures inspired by **quantum dynamics** and **biased diffusion**, designed to enhance GNN performance on tasks requiring the propagation of distant node information.

## Motivation

Traditional GNNs often struggle to propagate information across long distances due to oversmoothing and limited receptive fields. By leveraging concepts from quantum dynamics and learnable geometric transformations, our architectures aim to:

- Preserve signal across multiple hops.
- Allow flexible, learnable propagation that adapts to the learned task.
- Maintain stability through unitary operators and diffusion-inspired mechanisms.

## Methods

1. **Quantum-inspired GNN**  
   Uses learnable unitary operators to propagate node features, mimicking quantum state evolution. This allows long-range interactions without information loss.

2. **Biased Diffusion GNN**  
   Implements learnable biased diffusion on the graph, enabling the network to preferentially propagate features along important paths.

## Experiments

Run experiments to evaluate all models on a chosen dataset:

```bash
python experiments.py --dataset DATASET_NAME
