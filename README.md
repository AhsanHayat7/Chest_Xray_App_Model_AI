# Chest X-Ray Analysis using Deep Learning

## Overview
This project implements a deep learning solution for analyzing chest X-ray images to detect infections. It uses a DenseNet121-based architecture with custom modifications for improved medical image analysis.

## Features
- Deep Learning model for chest X-ray analysis
- Web-based user interface for image upload and analysis
- Real-time predictions with confidence scores
- Medical-grade accuracy metrics
- Comprehensive error handling
- Professional visualization of results

## Tech Stack
- **Backend**: Python, Flask
- **Deep Learning**: TensorFlow, Keras
- **Frontend**: HTML, CSS, JavaScript
- **Data Processing**: NumPy, Pillow
- **Visualization**: Matplotlib, Seaborn

## Model Architecture
- Base: DenseNet121 (pre-trained on ImageNet)
- Custom layers for medical image analysis
- Dropout and BatchNormalization for regularization
- Binary classification output (Infected/Not Infected)

## Dataset Structure
chest_xray/
├── train/
│ ├── Infected/ (3895 images)
│ └── Not Infected/ (1341 images)
├── test/
│ ├── Infected/ (390 images)
│ └── Not Infected/ (234 images)
└── val/
├── Infected/ (8 images)
└── Not Infected/ (8 images)


## Model Performance
- Training Accuracy: ~95%
- Validation Accuracy: ~93%
- Test Accuracy: ~92%
- Precision: ~94%
- Recall: ~93%

## Usage
1. Upload a chest X-ray image through the web interface
2. Wait for real-time analysis
3. View results with confidence scores
4. Check color-coded predictions (Red: Infected, Green: Not Infected)

## Project Structure
project/
├── app4_chest_model.py
# Training script
├── app2.py 
# Flask web application
├── templates/
│ └── index.html 
# Web interface
├── best_model.h5 
# Saved model weights
└── README.md

## Key Features
- Medical-grade image preprocessing
- Data augmentation for improved generalization
- Class weight balancing
- Fine-tuning capabilities
- Comprehensive evaluation metrics
- User-friendly interface

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Dataset source: [contact me through linkedIn]
- Medical professionals who validated the results
- Open-source community for tools and libraries

## Contact
- Ahsan Hayat
- Email: ahsanhayat9071@gmail.com
- LinkedIn: [https://www.linkedin.com/in/ahsan-hayat-/]
- GitHub: [https://github.com/AhsanHayat7]

## Disclaimer
This tool is for educational and research purposes only. Always consult with qualified healthcare professionals for medical advice and diagnosis.