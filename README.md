# CMI_Masters_Thesis

This repository contains the code and data for my Master's Thesis in Computer Science at Chennai Mathematical Institue. The thesis is titled "State Space Models: An Efficient Alternative to Attention" and is supervised by Professor Pranabendu Misra.

## Contents
- [CODE](./CODE/): Contains the code for the experiments.
- [PAPERS_READ](./PAPERS_READ/): Contains the papers read during the course of the thesis.
- [REPORT](./REPORT/): Contains the final report of the thesis.
- [SLIDES](./SLIDES/): Contains the slides for the presentation of the thesis.

## Abstract

![home](./home_pic.png)

The Transformer architecture, with its self-attention mechanism, has revolutionized deep learning but is computationally expensive for long sequences. This thesis surveys State Space Models (SSMs), including S4, Mamba, Linear Recurrent Units (LRUs), and Griffin, which offer efficiency and effectiveness in handling long-range dependencies. The thesis includes theoretical discussions, architectural explorations, and empirical results demonstrating the potential of SSMs in text, image, and audio generation tasks.

## Chapters Overview

### Chapter 1: Introduction to Deep Sequence Modelling

This chapter introduces deep sequence modelling, discussing the historical context and challenges, including the Long Range Arena (LRA) benchmark used to test long-range dependency handling.

### Chapter 2: Transformers and the Attention Mechanism

Provides an overview of the Transformer architecture and its attention mechanism, highlighting the limitations of attention in handling long sequences.

### Chapter 3: SSMs: Foundations and the S4 Architecture

Discusses the theoretical foundations of State Space Models and dives into the S4 architecture, explaining its components, training mechanisms, and performance in the LRA benchmark.

### Chapter 4: Selective SSM and Mamba

Explores the Mamba architecture, focusing on its selective state space and efficiency in processing long sequences.

### Chapter 5: Linear Recurrent Units: Simplicity and Universality

Introduces Linear Recurrent Units (LRUs), their theoretical universality, and their design to achieve performance comparable to SSMs with minimal changes.

### Chapter 6: Hybrid Models: Hawk and Griffin

Examines hybrid models that combine LRUs with local attention, achieving higher performance and efficiency, including Hawk and Griffin architectures.

### Chapter 7: Experiments

Presents experimental results showcasing the capabilities of SSMs on toy datasets, including text, image, and audio generation tasks.

## Installation

To run the code in this repository, you need to have Python 3.x installed. You can install the necessary dependencies using pip using the [requirements.txt](./CODE/requirements.txt) file:

```bash
pip install -r requirements.txt
```