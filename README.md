# LPCV 2023 Challenge - ENOT Pipeline

This document serves as the documentation for the solution, providing an overview of the training and inference model, as well as the ablation study conducted.

## Training
### Train Data
The training data consists of two components:

- Organizers' Data: This refers to the initial version of the dataset provided by the organizers. It includes the validation and train splits.

- [UAVid](https://arxiv.org/abs/1810.10438): During the training experiments, it was discovered that incorporating UAVid significantly improved the accuracy of the model, particularly for the person class.

### Model
As a starting point, the PIDNet model was used as a baseline. The PIDNet model is known for its good balance between speed and quality. [[paper](https://arxiv.org/pdf/2206.02066.pdf), [code](https://github.com/XuJiacong/PIDNet)]

However, a modification was made to the model architecture. The last convolutional layer along with the resizing operation was replaced with a transposed convolution. This change was found to improve both the accuracy of the model and the speed of the pipeline.

## Inference
### Postprocessing
The postprocessing step involves applying several heuristics based on observations of model predictions, real data, and validation metrics. The main essence of the postprocessing heuristics is as follows:

1. Cracks/Fissures/Subsidence: If cracks, fissures, or subsidence are detected in the predicted image, the pixels predicted as lava_flow are assigned to the background class. It was observed that the lava_flow class did not occur when these anomalies were present, and the model often made mistakes in such cases. Hence, this heuristic was proposed to address this issue.

2. Class Occupancy: If any class occupies less than 500 pixels in the image, the pixels of that class are assigned to the background. This heuristic was developed after analyzing the behavior of metrics based on the model's predictions. It was found that it is better to exclude predictions where a class occupies a small area, rather than predict the wrong class. Therefore, a filter was implemented to remove predictions with minimal class occupancy.

## Ablation Study
An ablation study was conducted to assess the impact of certain modifications on the model's performance.

- Replacement of the Last Convolutional Layer + Resize: The last convolutional layer along with the resizing operation was replaced with a transposed convolution. This modification resulted in a 2% increase in accuracy and improved the speed of the pipeline.

- Addition of Postprocessing: The implementation of postprocessing measures further improved the accuracy by 2.8%.
