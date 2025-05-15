# import cv2
# import torch
# import numpy as np
# from facenet_pytorch import MTCNN, InceptionResnetV1
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# import os

# # Device setup
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Initialize MTCNN and FaceNet
# mtcnn = MTCNN(keep_all=True, device=device)
# resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# # Face Classifier
# class FaceClassifier(nn.Module):
#     def __init__(self, input_dim=512, num_classes=5):
#         super(FaceClassifier, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(input_dim, 64),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(64, num_classes),
#             nn.Softmax(dim=1)
#         )
#     def forward(self, x):
#         return self.fc(x)

# classifier = FaceClassifier().to(device)

# # Load trained model
# def load_model(model_path):
#     checkpoint = torch.load(model_path, map_location=device)
#     resnet.load_state_dict(checkpoint['resnet_state_dict'])
#     classifier.load_state_dict(checkpoint['classifier_state_dict'])
#     resnet.eval()
#     classifier.eval()

# # Align face
# def align_face(image, landmarks):
#     left_eye = landmarks[0]
#     right_eye = landmarks[1]
#     dY = right_eye[1] - left_eye[1]
#     dX = right_eye[0] - left_eye[0]
#     angle = np.degrees(np.arctan2(dY, dX))
#     eyes_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
#     M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
#     aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
#     return aligned

# # Process frame
# def process_frame(frame, classes):
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     boxes, _, landmarks = mtcnn.detect(frame_rgb, landmarks=True)
    
#     if boxes is not None and len(boxes) > 0:
#         # Get bounding box
#         x, y, x2, y2 = boxes[0]
#         width = x2 - x
#         height = y2 - y
#         box_size = (width, height)
        
#         # Extract and process face
#         face = frame_rgb[int(y):int(y2), int(x):int(x2)]
#         if face.size == 0:
#             return None, None, None, None
        
#         face = cv2.resize(face, (160, 160))
#         face = align_face(face, landmarks[0])
#         face_tensor = torch.tensor(face).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        
#         # Get embedding and classify
#         embedding = resnet(face_tensor)
#         output = classifier(embedding)
#         probs = output.detach().cpu().numpy()[0]
#         identity = classes[np.argmax(probs)]
#         confidence = probs[np.argmax(probs)] * 100
        
#         # Draw bounding box on frame
#         frame = cv2.rectangle(frame, (int(x), int(y)), (int(x2), int(y2)), (0, 255, 0), 2)
        
#         return frame, identity, confidence, box_size
#     return frame, None, None, None


import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms
import random

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize MTCNN with adjusted parameters
mtcnn = MTCNN(keep_all=True, device=device, min_face_size=30, thresholds=[0.6, 0.7, 0.8])

# Initialize FaceNet
resnet = InceptionResnetV1(pretrained='vggface2').to(device)

# Data augmentation
data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Face Dataset
class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['abhinav', 'andy', 'adarsh', 'adam', 'unknown_subjects']
        self.data = []
        for idx, cls in enumerate(self.classes):
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.exists(cls_dir):
                print(f"Warning: Directory {cls_dir} does not exist!")
                continue
            for img_name in os.listdir(cls_dir):
                self.data.append((os.path.join(cls_dir, img_name), idx))
        
        # Shuffle and split for train/val
        random.shuffle(self.data)
        self.train_data = self.data[:int(0.8 * len(self.data))]
        self.val_data = self.data[int(0.8 * len(self.data)):]
        
        print(f"Training dataset: {len(self.train_data)} images")
        print(f"Validation dataset: {len(self.val_data)} images")

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        img_path, label = self.train_data[idx]
        img = cv2.imread(img_path)
        if img is None:
            return torch.zeros(3, 160, 160), -1
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Retry face detection with relaxed parameters
        boxes, _, landmarks = mtcnn.detect(img, landmarks=True)
        if boxes is None or len(boxes) == 0:
            mtcnn_temp = MTCNN(keep_all=True, device=device, min_face_size=20, thresholds=[0.5, 0.6, 0.6])
            boxes, _, landmarks = mtcnn_temp.detect(img, landmarks=True)
            if boxes is None or len(boxes) == 0:
                return torch.zeros(3, 160, 160), -1
        
        x, y, x2, y2 = boxes[0]
        x, y, x2, y2 = int(x), int(y), int(x2), int(y2)
        x, y, x2, y2 = max(0, x), max(0, y), min(img.shape[1], x2), min(img.shape[0], y2)
        
        if x2 <= x or y2 <= y:
            return torch.zeros(3, 160, 160), -1
        
        face = img[y:y2, x:x2]
        try:
            face = cv2.resize(face, (160, 160))
            face = align_face(face, landmarks[0])
            face_tensor = torch.tensor(face).permute(2, 0, 1).float() / 255.0
            if self.transform:
                face_tensor = self.transform(face)
            return face_tensor, label
        except:
            return torch.zeros(3, 160, 160), -1

# Align face
def align_face(image, landmarks):
    left_eye = landmarks[0]
    right_eye = landmarks[1]
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))
    eyes_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
    M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
    aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return aligned

# Face Classifier
class FaceClassifier(nn.Module):
    def __init__(self, input_dim=512, num_classes=5):
        super(FaceClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.fc(x)

classifier = FaceClassifier().to(device)

# Triplet Loss
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

triplet_loss = TripletLoss(margin=1.0)

# Load trained model
def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=device)
    resnet.load_state_dict(checkpoint['resnet_state_dict'])
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    resnet.eval()
    classifier.eval()

# Process frame with improved bounding box handling
last_detection = None  # Store last known detection
SMOOTHING_FRAMES = 5  # Number of frames to retain last detection
frame_count_since_detection = 0

def process_frame(frame, classes, confidence_threshold=0.7):
    global last_detection, frame_count_since_detection
    
    # Preprocess frame: enhance contrast
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.convertScaleAbs(frame_rgb, alpha=1.1, beta=5)  # Reduced contrast/brightness adjustment
    
    # Try primary MTCNN detection
    boxes, _, landmarks = mtcnn.detect(frame_rgb, landmarks=True)
    
    # Retry with more lenient parameters if detection fails
    if boxes is None or len(boxes) == 0:
        mtcnn_temp = MTCNN(keep_all=True, device=device, min_face_size=20, thresholds=[0.5, 0.6, 0.7])
        boxes, _, landmarks = mtcnn_temp.detect(frame_rgb, landmarks=True)
    
    if boxes is not None and len(boxes) > 0:
        # Use the original MTCNN bounding box coordinates
        x, y, x2, y2 = boxes[0]
        
        # Validate and constrain coordinates to frame bounds
        x, y, x2, y2 = int(max(0, x)), int(max(0, y)), int(min(frame.shape[1], x2)), int(min(frame.shape[0], y2))
        # if x2 <= x or y2 <= y:
        #     # Use last detection if recent
        #     if last_detection and frame_count_since_detection < SMOOTHING_FRAMES:
        #         frame_count_since_detection += 1
        #         return frame, last_detection['identity'], last_detection['confidence'], last_detection['box_size']
        #     frame_count_since_detection += 1
        #     return frame, None, None, None
        
        # width = x2 - x
        # height = y2 - y
        # box_size = (width, height)
        
        # Debug print
        # print(f"Detected box: x={x}, y={y}, x2={x2}, y2={y2}, width={width}, height={height}")
        
        face = frame_rgb[int(y):int(y2), int(x):int(x2)]
        if face.size == 0:
            # Use last detection if recent
            if last_detection and frame_count_since_detection < SMOOTHING_FRAMES:
                frame_count_since_detection += 1
                return frame, last_detection['identity'], last_detection['confidence'], last_detection['box_size']
            frame_count_since_detection += 1
            return frame, None, None, None
        
        face = cv2.resize(face, (160, 160))
        face = align_face(face, landmarks[0])
        face_tensor = torch.tensor(face).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        
        embedding = resnet(face_tensor)
        output = classifier(embedding)
        probs = output.detach().cpu().numpy()[0]
        max_confidence = np.max(probs)
        
        # Update last detection (without box_coords for drawing)
        # last_detection = {
        #     'identity': classes[np.argmax(probs)],
        #     'confidence': max_confidence * 100,
        #     'box_size': box_size
        # }
        # frame_count_since_detection = 0
        
        # if max_confidence < confidence_threshold:
        #     frame = cv2.rectangle(frame, (int(x), int(y), int(x2), int(y2)), (0, 255, 0), 2)
        #     return frame, "unknown_subjects", max_confidence * 100, box_size
        
        # identity = classes[np.argmax(probs)]
        # confidence = max_confidence * 100
        
        # frame = cv2.rectangle(frame, (int(x), int(y), int(x2), int(y2)), (0, 255, 0), 2)
        # return frame, identity, confidence, box_size
    
    # Use last detection if recent (without drawing box)
    if last_detection and frame_count_since_detection < SMOOTHING_FRAMES:
        frame_count_since_detection += 1
        return frame, last_detection['identity'], last_detection['confidence'], last_detection['box_size']
    
    frame_count_since_detection += 1
    return frame, None, None, None

# Training function
def train_model(dataset, model_path, epochs=50):
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(FaceDataset(dataset.root_dir, transform=None), batch_size=16, shuffle=False)
    
    optimizer = torch.optim.Adam(list(resnet.parameters()) + list(classifier.parameters()), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    ce_loss = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    for epoch in range(epochs):
        resnet.train()
        classifier.train()
        train_loss = 0
        for batch in train_loader:
            images, labels = batch
            if (labels == -1).any():
                continue
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            embeddings = resnet(images)
            outputs = classifier(embeddings)
            
            # Classification loss
            loss_ce = ce_loss(outputs, labels)
            
            # Triplet loss (select anchor, positive, negative)
            anchor = embeddings[0:1]
            positive = embeddings[1:2] if labels[1] == labels[0] else embeddings[2:3]
            negative = embeddings[2:3] if labels[2] != labels[0] else embeddings[1:2]
            loss_triplet = triplet_loss(anchor, positive, negative)
            
            loss = loss_ce + 0.5 * loss_triplet
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        resnet.eval()
        classifier.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch
                if (labels == -1).any():
                    continue
                images, labels = images.to(device), labels.to(device)
                embeddings = resnet(images)
                outputs = classifier(embeddings)
                loss = ce_loss(outputs, labels)
                val_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'resnet_state_dict': resnet.state_dict(),
                'classifier_state_dict': classifier.state_dict()
            }, model_path)

def classify_person(image, classes=['abhinav', 'andy', 'adarsh', 'adam', 'unknown_subjects'], confidence_threshold=0.7):
    """
    Classify a person in an image.
    
    Args:
        image (numpy.ndarray): Input image in BGR format (from cv2.imread or video frame).
        classes (list): List of class names (default: ['abhinav', 'andy', 'adarsh', 'adam', 'unknown_subjects']).
        confidence_threshold (float): Confidence threshold for classification (default: 0.7).
    
    Returns:
        tuple: (identity, confidence) where identity is the predicted class name (or None if no face detected),
               and confidence is the confidence score in percentage (or None if no face detected).
    """
    # Ensure image is in RGB format
    if image is None:
        return None, None
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Enhance contrast
    img_rgb = cv2.convertScaleAbs(img_rgb, alpha=1.1, beta=5)
    
    # Detect face
    boxes, _, landmarks = mtcnn.detect(img_rgb, landmarks=True)
    
    # Retry with more lenient parameters if detection fails
    if boxes is None or len(boxes) == 0:
        mtcnn_temp = MTCNN(keep_all=True, device=device, min_face_size=20, thresholds=[0.5, 0.6, 0.7])
        boxes, _, landmarks = mtcnn_temp.detect(img_rgb, landmarks=True)
    
    if boxes is None or len(boxes) == 0:
        print("No face detected in the image")
        return None, None
    
    # Extract face
    x, y, x2, y2 = boxes[0]
    if landmarks is not None and len(landmarks) > 0:
        left_eye = landmarks[0][0]
        right_eye = landmarks[0][1]
        eye_distance = np.linalg.norm(left_eye - right_eye)
        face_width = eye_distance * 2.5
        face_height = face_width * 1.2
        eyes_center_x = (left_eye[0] + right_eye[0]) / 2
        eyes_center_y = (left_eye[1] + right_eye[1]) / 2
        x = eyes_center_x - face_width / 2
        y = eyes_center_y - face_height / 2
        x2 = eyes_center_x + face_width / 2
        y2 = eyes_center_y + face_height / 2
    
    # Validate coordinates
    x, y, x2, y2 = int(max(0, x)), int(max(0, y)), int(min(img_rgb.shape[1], x2)), int(min(img_rgb.shape[0], y2))
    if x2 <= x or y2 <= y:
        print("Invalid face bounding box")
        return None, None
    
    face = img_rgb[int(y):int(y2), int(x):int(x2)]
    if face.size == 0:
        print("Empty face region extracted")
        return None, None
    
    # Preprocess face
    face = cv2.resize(face, (160, 160))
    face = align_face(face, landmarks[0])
    face_tensor = torch.tensor(face).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    
    # Classify
    with torch.no_grad():
        embedding = resnet(face_tensor)
        output = classifier(embedding)
        probs = output.cpu().numpy()[0]
        max_confidence = np.max(probs)
    
    if max_confidence < confidence_threshold:
        return "unknown_subjects", max_confidence * 100
    
    identity = classes[np.argmax(probs)]
    confidence = max_confidence * 100
    return identity, confidence