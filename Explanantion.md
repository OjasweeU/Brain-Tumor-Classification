**High level overview**

This project is an end to end deep learning pipeline for brain tumour classification from MRI images, combined with model explainability using Grad CAM.
The goal was not only to predict the tumour type, but also to visualise which regions of the MRI influenced the model’s decision, which is critical in medical AI.
The model classifies MRI scans into four mutually exclusive classes: glioma, meningioma, pituitary tumour, and no tumour.



**Dataset understanding and motivation**

I used the Brain Tumour MRI dataset from Kaggle, which contains labelled MRI images organised into training and testing folders.
Before training, I did three important checks:
  - First, I verified the folder structure and class names, since the directory names are directly used as labels by Keras generators.
  - Second, I visually inspected random images from each class to ensure the data quality and correct labelling.
  - Third, I checked class distribution to understand whether there was any major class imbalance that could bias training.

This step is important because in medical datasets, imbalance and mislabelling can significantly affect model reliability.



**Image preprocessing and brain region cropping**

A key part of my pipeline is explicit image preprocessing using OpenCV.
Instead of feeding raw MRI images directly to the network, I first cropped the brain region from each image.
The logic is as follows:
  - I converted the image to grayscale, applied Gaussian blur to reduce noise, and then used thresholding followed by erosion and dilation to isolate the main connected component.
  - I then extracted contours and selected the largest contour, assuming it corresponds to the brain.
  - From that contour, I computed extreme points on all four sides and cropped the original image accordingly.
  - This step removes background artifacts, borders, and text labels, and forces the model to focus only on anatomical features relevant to tumour detection.
  - After cropping, I resized all images to 240 × 240, ensuring consistent input dimensions across the dataset.
**I saved these processed images into a new directory structure, so the model trains on a clean, preprocessed dataset.**



**Data loading and augmentation strategy**

For model training, I used ImageDataGenerator.
I applied data augmentation only on the training data, including small rotations, horizontal flips, and vertical shifts.
This helps improve generalisation and reduce overfitting, which is important because medical datasets are usually limited in size.
I also created a validation split from the training set, ensuring that validation data is never augmented.
The test set was kept completely unseen during training and used only for final evaluation.
Each image is automatically labelled based on its folder name, and labels are generated in one hot encoded format, since this is a multi class classification problem.



**Model selection and architecture**

For the model, I used EfficientNetB1 with transfer learning.
EfficientNet was chosen because it provides an excellent balance between accuracy and computational efficiency by scaling depth, width, and resolution together.
I loaded the pretrained EfficientNet backbone with ImageNet weights and removed its original classification head.
On top of this backbone, I added:
  - A global pooling layer to aggregate spatial features
  - A dropout layer for regularisation and overfitting control
  - A dense softmax layer with four outputs, corresponding to the four tumour classes
This design allows the model to reuse low level and mid level visual features learned from ImageNet and adapt them to the medical imaging domain.



**Training configuration and optimisation**

I compiled the model using the Adam optimiser with a small learning rate to ensure stable fine tuning.
The loss function used was categorical cross entropy, which is appropriate because:
  - The classes are mutually exclusive
  - Labels are one hot encoded
  - The output layer uses softmax activation
To make training robust, I used three callbacks:
  - Model checkpointing to save the best performing model based on validation accuracy
  - Early stopping to prevent overfitting when validation performance stops improving
  - Reduce learning rate on plateau to fine tune the model when training stagnates
    
_Training was done for up to 50 epochs, but early stopping ensured that training terminated once the model converged._



**Model evaluation and performance analysis**

After training, I evaluated the model on both training and test sets.
Beyond accuracy, I used a confusion matrix and classification report to analyse per class performance.
This helped identify which tumour types were most frequently confused with each other, which is especially important in medical applications.
I also performed qualitative evaluation by randomly selecting test images and comparing predicted labels with true labels to ensure predictions made clinical sense.



**Model explainability using Grad CAM**

To add explainability, I implemented Grad CAM.
Grad CAM works by computing the gradient of the predicted class score with respect to the feature maps of the last convolution layer.
These gradients indicate which spatial regions contributed most to the model’s prediction.
The process is:
  - Compute gradients using TensorFlow’s GradientTape
  - Average gradients across spatial dimensions to get channel importance
  - Create a weighted sum of feature maps
  - Apply ReLU to keep only positive evidence
  - Upsample and overlay the heatmap on the original MRI image
    
_The resulting heatmap visually highlights tumour regions that influenced the prediction.
This step is crucial because in healthcare, predictions without interpretability are not trustworthy._



**Key limitations and improvements**

If I were to extend this project further, I would:
  - Apply the official EfficientNet preprocessing function to ensure optimal feature scaling
  - Explicitly freeze and then gradually unfreeze backbone layers during fine tuning
  - Add class weighted loss if imbalance becomes severe
  - Validate Grad CAM results with clinical annotations if available
