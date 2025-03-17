# Face Recognition Project

This project is a Python-based real-time Face Recognition system that uses OpenCV, MediaPipe, and a machine learning model to detect and recognize faces. Users must collect data, process it, train the model, and then test it for face recognition.

## Features
- Real-time face detection using MediaPipe Face Detection.
- Face landmark mapping with MediaPipe Face Mesh.
- Accurate face recognition powered by a user-trained machine learning model.
- Easily extensible for adding new individuals or improving recognition accuracy.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/rishraks/Face_Recognition.git
   cd face-recognition-project
   ```

2. **Set Up Virtual Environment:**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install mediapipe opencv-python
   ```

4. **Organize Data for Training:**
   - Create a folder `Data_Collection` with subfolders named after each person's name.
   - Place corresponding images in each folder.

5. **Train the Model:**
   - Run the training script:
     ```bash
     python Data_Training.py
     ```
   - Save the generated model as `model.p` in the project directory.

## Usage

### Real-Time Face Recognition
After training the model, run the following command to start the application:
```bash
python Data_Testing.py
```
- Press `Q` to quit the application.

### Training a New Model
To train a new model with your dataset:
1. Ensure the dataset is organized in the `Data_Collection` folder.
2. Run the training script:
   ```bash
   python training.py
   ```
3. Save the generated model as `model.p`.

## Requirements
- Python 3.7+
- OpenCV
- MediaPipe
- NumPy
- Scikit-learn

## File Structure
```
face-recognition-project/
├── Data_Collection/       # Dataset folder with subfolders for each person
├── model.p                # Trained model (generated by the user)
├── Testing.py             # Script for real-time face recognition
├── Data_Training.py       # Script for training the face recognition model
├── Data_Collection.py     # Script for data collection
├── Data_Processing.py     # Script for data processing
└── README.md              # Project documentation
```

## Future Enhancements
- Add support for multiple cameras.
- Improve recognition accuracy with more training data.
- Implement GUI for user-friendly interaction.

## Acknowledgements
This project leverages the following libraries and tools:
- [OpenCV](https://opencv.org/)
- [MediaPipe](https://mediapipe.dev/)
- [Scikit-learn](https://scikit-learn.org/)

## License
This project is licensed under the MIT License. See `LICENSE` for more details.
