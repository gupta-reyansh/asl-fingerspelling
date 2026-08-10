# ASL Fingerspelling Translation with a Hybrid CNN-Transformer Model

A real-time American Sign Language (ASL) fingerspelling translation system using a hybrid CNN-Transformer deep learning model developed by me for my 8th grade science fair project. This project uses the MediaPipe library for hand and face marker detection combined with a TensorFlow Lite model for quick predictions.

## Features

- **Real-time Recognition**: Captures and recognizes ASL fingerspelling through your webcam
- **Auto-correction**: Implements spell-checking to correct predicted text using the autocorrect library

## Project Structure

```markdown
asl-fingerspelling/
├── Scripts/
│   └── program.py                       # Main inference script for real-time recognition
├── Data/
│   ├── character_to_prediction_index.json  # Character to model output mapping
│   ├── inference_args.json              # Inference configuration
│   └── requirements.txt                 # Python dependencies
├── Model.tflite                         # Pre-trained TensorFlow Lite model
├── LICENSE                              # Project license
└── README.md                            # This file
```

## Requirements

For a complete list of dependencies, see [Data/requirements.txt](Data/requirements.txt).

## Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd asl-fingerspelling
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r Data/requirements.txt
   ```

## Usage

### Real-time Recognition

Run the main inference script to start the translation:

```bash
python Scripts/program.py
```

**Controls:**

- Press 'SPACE' to start capturing frames
- Press 'ESC' to stop capturing

### How It Works

1. **Hand Detection**: Uses the MediaPipe library to detect hand, face, and body landmarks
2. **Feature Extraction**: Extracts relevant features from left and right hands, as well as facial features
3. **Model Inference**: Processes the features through the pre-trained Hybrid CNN-Transformer TensorFlow Lite model
4. **Prediction**: Returns the predicted character sequence
5. **Spell Correction**: Applies auto-correction to improve prediction accuracy

### Configuration

The script can be customized by modifying parameters in `Scripts/program.py`:

- **Webcam dimensions**: `cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)` and `cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)`
- **Detection confidence**: Adjust `min_detection_confidence` and `min_tracking_confidence` in the Holistic configuration
- **Frames to capture**: Change `FRAMES_TO_CAPTURE` variable (currently set to 768)

## Model Details

- **Architecture**: Hybrid CNN-Transformer
- **Input Format**: Holistic landmark features (hand positions, pose, facial features)
- **Output Format**: Character predictions for ASL fingerspelling
- **Optimization**: TensorFlow Lite format for efficient inference

The pre-trained model is stored as `Model.tflite` and uses the "serving_default" signature.

## Key Components

### Hand Landmarks

- **LEFT**: Left hand and left-side features (468-488 indices)
- **RIGHT**: Right hand and right-side features (522-542 indices)
- **CENTRE**: Center body features (face and center pose)

### Data Files

- **character_to_prediction_index.json**: Maps predicted indices to ASL characters
- **inference_args.json**: Stores inference-time arguments and configurations

## Performance

The system is optimized for real-time performance:

- Processes video at resolution: 1280×720
- Uses TensorFlow Lite for efficient inference
- Minimum detection confidence: 0.5
- Frame-based processing with 768-frame sequences

## Future Improvements

- [ ] Support for continuous fingerspelling recognition (without frame capture mode)
- [ ] Web interface for easy access
- [ ] Support for additional sign languages

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Contact

For questions or inquiries about this project, please reach out to ryetheguy93@gmail.com 
Feel free to submit a pull request or an issue, and I will do my best to help
