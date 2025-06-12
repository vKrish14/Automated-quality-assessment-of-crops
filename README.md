Overview

This project demonstrates an end-to-end pipeline for automated crop quality assessment using a Convolutional Neural Network (CNN) trained on a synthetic PlantVillage-style dataset. The code is designed for Google Colab and includes all steps—from dataset generation to model training, evaluation, and inference.
Features

    Synthetic Dataset Generation: Creates a dataset with two classes, "Healthy" and "Diseased," each containing 100 images of synthetic leaves. Diseased leaves feature random dark spots to simulate disease symptoms.

    Data Augmentation: Uses Keras' ImageDataGenerator for real-time data augmentation during training.

    CNN Model: Implements a simple yet effective CNN architecture for image classification.

    Visualization: Plots training and validation accuracy/loss curves.

    Model Saving: Saves the trained model for future inference.

    Inference Function: Provides a utility to predict the class of a new leaf image.

Requirements

    Google Colab environment (recommended)

    Python 3.x

    Libraries: tensorflow, opencv-python, tqdm, numpy, matplotlib, Pillow

All dependencies are installed automatically within the code.
How to Use

    Open the notebook in Google Colab.

    Run the entire code block.

        The script will generate a synthetic dataset at /content/plantvillage.

        It will train a CNN on this dataset, evaluate its performance, and plot results.

        The trained model will be saved as crop_quality_cnn.h5.

    Inference:

        Use the provided predict_image function to classify new images.

        Example usage is included at the end of the code (uncomment to use).

Dataset Structure

After running the code, your dataset will be organized as:

text
/content/plantvillage/
    ├── Healthy/
    │     ├── healthy_0.jpg
    │     ├── healthy_1.jpg
    │     └── ...
    └── Diseased/
          ├── diseased_0.jpg
          ├── diseased_1.jpg
          └── ...

Customization

    Number of Images: Change images_per_class in the code to generate more or fewer images per class.

    Image Size: Adjust img_height and img_width as needed.

    Model Complexity: Modify the CNN architecture for more complex tasks or larger datasets.

    Training Epochs: Increase epochs for better accuracy.

Notes

    This synthetic dataset is for demonstration and prototyping. For real-world applications, use actual crop images.

    The code and dataset are fully compatible with Google Colab and require no manual setup.

Author:
Adapted for educational and demonstration purposes.
For questions or improvements, please open an issue or submit a pull request.
