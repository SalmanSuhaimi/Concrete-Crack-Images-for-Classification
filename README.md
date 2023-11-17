# Concrete Crack Images for Classification
## Project Description

This project utilizes a dataset comprising 40,000 concrete images, sourced from METU Campus Buildings, divided into two categories: 20,000 images with cracks (positive) and 20,000 without cracks (negative). Each image is sized 227 x 227 pixels with RGB channels. The developed AI model employs transfer learning for image classification, distinguishing between cracked and crack-free concrete. The outcome guides prioritized maintenance and repairs, enhancing overall structural safety.

## Problem Statement:
- Cracked buildings pose a severe threat to public safety due to their potential for structural failure.
- These cracks compromise both the stability of the building and the safety of occupants.
- The unpredictable nature of crack propagation makes assessing the extent of damage challenging, creating a hazardous environment.
- Addressing cracked buildings is crucial to prevent catastrophic events and ensure the safety and resilience of our infrastructure.

<p align="center">
<img src="https://github.com/SalmanSuhaimi/Concrete-Crack-Images-for-Classification/assets/148429853/56147c9d-515f-42d1-89f2-f2b995cc2bbc" width="500" height="500"/>
</p>

## Objective:
- The objective of using transfer learning is to optimize the AI model for concrete image classification.
- By harnessing knowledge from the broader METU campus dataset, the model aims to better recognize patterns and features relevant to crack detection.
- The goal is to enhance the model's efficiency, enabling it to accurately identify cracks in concrete images, contribute to maintenance prioritization, and ultimately improve overall structural safety.

### Why do We Use Transfer Learning?
- We use transfer learning to find cracks in buildings because it's like using a smart helper that already knows a lot about pictures.
- This helper has learned from many different images before.
- We teach it specifically about cracked and non-cracked buildings by showing it our pictures. This way, it quickly learns what cracks look like.
- Transfer learning is like giving our helper a head start, making it better at spotting cracks in buildings without starting from scratch.

### Sample Data Collection:
<p align="center">
<img src="https://github.com/SalmanSuhaimi/Concrete-Crack-Images-for-Classification/assets/148429853/9aa9a0c5-a88f-4e40-afa8-2b89cd6adec6" width="500" height="500"/>
</p>

The picture above shows some sample data collected from METU Campus Buildings. Every crack picture has been labeled either positive (crack) or negative (not crack).

### Flow Transfer Learning
I use a method of transfer learning implemented using a pre-trained MobileNetV2 model for image classification. 

- A dataset containing images of cracked and non-cracked buildings is loaded and split into training and validations 

- Data augmentation techniques are applied to the training dataset to increase its diversity. 
<p align="center">
<img src="https://github.com/SalmanSuhaimi/Concrete-Crack-Images-for-Classification/assets/148429853/75f83de9-7584-4575-829c-dd3f39793110)" width="500" height="500"/>
</p>
The primary benefits include increased model generalization, as it learns to recognize objects in various orientations and conditions, reduced overfitting by introducing variability, and improved performance in scenarios with limited labeled data.

- The MobileNetV2 model is then loaded as a feature extractor, and its layers are frozen to prevent further training. 

- A custom classification head is added on top of the feature extractor, including a global average pooling layer and a dense output layer. 

- The model is compiled with an Adam optimizer and sparse categorical crossentropy loss.
  
- The model is then trained on the training dataset, and the training progress is visualized using TensorBoard.
![epoch_accuracy](https://github.com/SalmanSuhaimi/Concrete-Crack-Images-for-Classification/assets/148429853/59e79668-d6a1-486a-acbd-86a8b876f441)
The picture shows accurate data during 10 epochs. The graph shows a good model
  
- After training, the model is fine-tuned by unfreezing some of the layers of the feature extractor and retraining on the dataset.
  
- Finally, the model is evaluated on the test dataset, and a batch of test images is used to demonstrate predictions.
<p align="center">
<img src="https://github.com/SalmanSuhaimi/Concrete-Crack-Images-for-Classification/assets/148429853/f62c9233-b2b4-48fc-bda7-26c8aace4f29" width="600" height="600"/>
<p>
The picture shows data prediction whether the image is positive or negative

## Conclusion
In conclusion, the application of transfer learning has proven to be highly effective in the task of crack detection. Leveraging pre-trained models on large datasets, especially in image recognition tasks, provides a powerful foundation for learning intricate features that are relevant to crack identification. By fine-tuning a pre-trained model on a specific crack detection dataset, the model can adapt and specialize, yielding impressive results. This approach not only capitalizes on the knowledge gained from broader datasets but also significantly reduces the need for extensive labeled data in the target domain. Transfer learning, therefore, emerges as a robust and efficient methodology for achieving accurate and reliable crack detection results.

## Data source:
https://data.mendeley.com/public-files/datasets/5y9wdsg2zt/files/8a70d8a5-bce9-4291-bab9-b48cfb3e87c3/file_downloaded
