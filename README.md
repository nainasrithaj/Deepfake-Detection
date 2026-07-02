# Deepfake-Detection
AI-powered deepfake detection using NVIDIA Jetson, PyTorch, and OpenCV. This project classifies images as real or AI-generated to help combat misinformation, identity fraud, and the spread of fake nude images by detecting manipulated content before it can be shared.


Deepfake Detection Using ResNet-18 (Jetson Inference + TensorRT)

This project classifies facial images as either real or fake (deepfake) using a re-trained ResNet-18 deep learning model deployed on NVIDIA Jetson devices.

The model is optimized with TensorRT and runs efficiently on edge hardware, making it suitable for real-time or offline deepfake detection in low-power environments.


The Algorithm

This system uses a re-trained ResNet-18 convolutional neural network trained on a dataset of real and deepfake facial images.

The model was exported to ONNX format and executed using NVIDIA’s imagenet.py tool from the Jetson Inference framework.

Output Classes
real
fake

The model performs image classification by analyzing facial features and predicting whether the image is authentic or AI-generated.

Project Structure
jetson-inference/
└── python/
    └── training/
        └── classification/
            └── models/
                └── deepfake_detection/
                    ├── resnet18.onnx
                    ├── labels.txt
                    ├── model_best.pth.tar
                    ├── checkpoint.pth.tar
                    └── tensorboard/


The dataset is organized using the standard ImageFolder format:

real-vs-fake/
├── train/
│   ├── real/
│   └── fake/
└── valid/
    ├── real/
    └── fake/



Running Inference (Image Classification)

Navigate to the classification directory:

cd ~/jetson-inference/python/training/classification


Run the model on a single image
python3 ~/jetson-inference/python/examples/imagenet.py \
  --model=models/deepfake_detection/resnet18.onnx \
  --labels=models/deepfake_detection/labels.txt \
  --input_blob=input_0 \
  --output_blob=output_0 \
  --input=/home/nvidia/datasets/deepfake/real_vs_fake/real-vs-fake/valid/real/68375.jpg \
  --output=output.jpg

  This project demonstrates how a lightweight deep learning model can be deployed on edge devices using NVIDIA Jetson and TensorRT for real-time image classification of real vs fake facial images.

  This is the link to my video:
