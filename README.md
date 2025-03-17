# Parallel Computing for Enhanced Pneumonia Detection in X-Rays

This repository contains the code and documentation for a project that leverages parallel computing techniques to enhance the efficiency of pneumonia detection from chest X-ray images using deep learning.

## Table of Contents

- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Methodology and Approach](#methodology-and-approach)
  - [EDA Using Multiprocessing](#eda-using-multiprocessing)
  - [Model Selection: DenseNet-161](#model-selection-densenet-161)
  - [Model Training using Distributed Data Parallel (DDP)](#model-training-using-distributed-data-parallel-ddp)
  - [Model Evaluation using Dask](#model-evaluation-using-dask)
- [Analysis and Results](#analysis-and-results)
- [Conclusion](#conclusion)
- [References](#references)

## Introduction

Pneumonia is a life-threatening lung infection that demands rapid and accurate diagnosis, especially for vulnerable populations such as children and the elderly. Traditional diagnosis via manual X-ray interpretation is both time-consuming and prone to variability. This project aims to address these challenges by applying deep learning techniques combined with parallel computing to accelerate data processing, model training, and evaluation.

## Problem Statement

The primary objective of this project is to develop a deep learning model capable of classifying chest X-ray images into two categories: pneumonia and normal. Given the large number of high-resolution images available (~5,863 images from a publicly available Kaggle dataset), significant computational resources are required. This project employs parallel computing strategies to:
- **Speed up data preprocessing and EDA**: Utilizing multiprocessing to load and process thousands of images in parallel.
- **Enhance model training efficiency**: Leveraging Distributed Data Parallel (DDP) on multiple GPUs.
- **Accelerate evaluation**: Using Dask to distribute the workload during model testing.

## Exploratory Data Analysis (EDA)

The EDA phase was critical for understanding the dataset and ensuring data quality before training. Key steps included:

- **Dataset Overview**: The Pediatric Pneumonia Chest X-ray dataset comprises 5,863 grayscale X-ray images, categorized as 'normal' or 'pneumonia'.
- **Data Preprocessing Challenges**: Loading high-quality images sequentially proved to be time-consuming. To overcome this, Python’s `multiprocessing.Pool` was employed to parallelize the image loading process.
- **Performance Improvement**: Testing was conducted across varying numbers of CPU cores (1, 2, 4, 8, and 16). Results indicated a significant reduction in processing time when increasing the number of CPUs, with diminishing returns observed beyond a certain point.
  
*The detailed performance analysis of EDA, including elapsed times and speedup graphs, can be found in the project report.*

## Methodology and Approach

### EDA Using Multiprocessing

- **Objective**: Reduce the time required to load and preprocess ~6000 high-resolution images.
- **Method**: Implemented the `multiprocessing.Pool` library in Python to distribute the workload across multiple CPU cores.
- **Outcome**: Achieved faster data loading, which is critical for timely exploratory analysis and subsequent model training.

### Model Selection: DenseNet-161

- **Why DenseNet-161?**
  - **Deep Architecture**: The 161-layer model efficiently captures complex features necessary for accurate image classification.
  - **Densely Connected Layers**: These connections help mitigate the vanishing gradient problem, enhance feature reuse, and reduce the total number of parameters.
- **Application**: Transfer learning was used to fine-tune DenseNet-161 for the pneumonia detection task.

### Model Training Using Distributed Data Parallel (DDP)

- **Purpose**: Speed up the training process by distributing the workload across multiple GPUs.
- **Approach**:
  - The dataset was partitioned, and the model was replicated across available GPUs.
  - Synchronized gradient updates ensured consistency during training.
- **Observations**:
  - Training times varied with batch size and number of GPUs.
  - Significant speedup was achieved with 4 GPUs, especially at larger batch sizes.
  
### Model Evaluation Using Dask

- **Goal**: Efficiently evaluate model performance on test data.
- **Technique**: Implemented Dask to parallelize the evaluation process by distributing the computation across multiple workers.
- **Benefits**:
  - Reduced evaluation time.
  - Scalable approach to handle large test datasets.

## Analysis and Results

- **EDA Performance**: Parallel processing significantly decreased the image loading time. The analysis highlighted the optimal number of CPU cores for efficient processing.
- **Training Efficiency**:
  - **Single vs. Multi-GPU Setups**: Training on 4 GPUs with DDP yielded the best performance improvement, with training times decreasing as batch sizes increased.
  - **Speedup and Efficiency Metrics**: The experiments indicated that while adding GPUs reduces overall training time, the efficiency (speedup per GPU) exhibits diminishing returns beyond an optimal point.
- **Model Performance**:
  - **Accuracy**: The model achieved high training and validation accuracies (~96%), indicating robust learning.
  - **Loss Convergence**: Training and validation loss curves converged, suggesting minimal overfitting.
  - **Evaluation Metrics**: The confusion matrix and precision-recall curves further confirmed the model's capability to distinguish between pneumonia and normal cases effectively.
- **Evaluation with Dask**: The distributed evaluation using Dask resulted in significant time savings without compromising accuracy.

## Conclusion

This project demonstrates that integrating parallel computing techniques can substantially improve the efficiency of deep learning workflows in medical imaging applications. Key takeaways include:

- **Data Loading**: Multiprocessing significantly reduced preprocessing time.
- **Model Training**: Distributed Data Parallel (DDP) effectively leveraged multiple GPUs, enabling faster training times and better scalability.
- **Model Evaluation**: Dask provided an efficient solution for evaluating model performance on large datasets.
- **Overall Impact**: The combination of these techniques led to a robust and scalable pneumonia detection system, capable of delivering high accuracy while reducing computational overhead.

## References

- Python Multiprocessing Documentation: [docs.python.org/3/library/multiprocessing.html](https://docs.python.org/3/library/multiprocessing.html)
- Dask Documentation: [docs.dask.org](https://docs.dask.org/en/stable/)
- PyTorch Distributed Data Parallel: [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- DenseNet-161 and Transfer Learning: Various sources on deep learning for medical imaging.
- Pediatric Pneumonia Chest X-ray Dataset on Kaggle
