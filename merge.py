import cv2
import mediapipe as mp
import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_dir = "Data_Collection"
os.makedirs(data_dir, exist_ok=True)

def collect_data(person_name=None):
    cap = cv2.VideoCapture(0)
    if not person_name:
        person_name = input("Enter your name: ").strip()
    person_dir = os.path.join(data_dir, person_name)
    os.makedirs(person_dir, exist_ok=True)
    
    dataset_size = 500
    count = 0
    
    while count < dataset_size:
        ret, frame = cap.read()
        if not ret:
            break
        img_path = os.path.join(person_dir, f"{person_name}_{count}.png")
        cv2.imwrite(img_path, frame)
        count += 1
        cv2.imshow("Collection", cv2.resize(frame, (800, 700)))
        if cv2.waitKey(1) == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def extract_features():
    mp_face_mesh = mp.solutions.face_mesh
    data, labels = [], []
    
    with mp_face_mesh.FaceMesh(max_num_faces=2, min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_landmarks=True) as mesh:
        for dir_ in os.listdir(data_dir):
            for img_path in os.listdir(os.path.join(data_dir, dir_)):
                data_aux = []
                img = cv2.imread(os.path.join(data_dir, dir_, img_path))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = mesh.process(img_rgb)
                
                if results.multi_face_landmarks:
                    for landmarks in results.multi_face_landmarks:
                        for i in range(len(landmarks.landmark)):
                            x, y = landmarks.landmark[i].x, landmarks.landmark[i].y
                            data_aux.extend([x, y])
                    data.append(data_aux)
                    labels.append(dir_)
    
    with open('data.pickle', 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)

def train_model():
    with open('data.pickle', 'rb') as f:
        data_dict = pickle.load(f)
    
    data, labels = np.array(data_dict['data']), np.array(data_dict['labels'])
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
    
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    
    y_predict = model.predict(x_test)
    score = accuracy_score(y_predict, y_test)
    print("Model Accuracy:", score)
    
    with open('model.p', 'wb') as f:
        pickle.dump({'model': model}, f)

def real_time_recognition():
    cap = cv2.VideoCapture(0)
    model_dict = pickle.load(open("model.p", "rb"))
    model = model_dict["model"]
    labels_dict = {folder: folder for folder in os.listdir(data_dir)}
    
    mp_face_mesh = mp.solutions.face_mesh
    mp_detect_face = mp.solutions.face_detection
    
    with mp_face_mesh.FaceMesh(max_num_faces=2, min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_landmarks=True) as mesh:
        with mp_detect_face.FaceDetection(model_selection=0, min_detection_confidence=0.6) as detect:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                H, W, _ = frame.shape
                
                data_aux = []
                output = detect.process(frame)
                if output.detections:
                    for detections in output.detections:
                        bbox = detections.location_data.relative_bounding_box
                        x1, y1, w, h = int(bbox.xmin * W), int(bbox.ymin * H), int(bbox.width * W), int(bbox.height * H)
                        cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)
                
                results = mesh.process(frame)
                if results.multi_face_landmarks:
                    for landmarks in results.multi_face_landmarks:
                        for i in range(len(landmarks.landmark)):
                            x, y = landmarks.landmark[i].x, landmarks.landmark[i].y
                            data_aux.extend([x, y])
                    
                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_person = labels_dict.get(prediction[0], "Unknown")
                    
                    if predicted_person == "Unknown":
                        print("Face not recognized. Collecting new data...")
                        new_name = input("Enter your name: ").strip()
                        collect_data(new_name)
                        extract_features()
                        train_model()
                        continue
                    
                    cv2.putText(frame, predicted_person, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                cv2.imshow("Testing", frame)
                if cv2.waitKey(1) == ord("q"):
                    break
    cap.release()
    cv2.destroyAllWindows()

if os.path.exists("model.p"):
    real_time_recognition()
else:
    print("No existing model found. Starting data collection...")
    collect_data()
    extract_features()
    train_model()
    real_time_recognition()