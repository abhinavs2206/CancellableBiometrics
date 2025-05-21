import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms
import random
from PIL import Image
from piq import ssim, psnr
import hmac
import hashlib

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
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

class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['abhinav', 'adarsh', 'arshini', 'imposters']
        self.data = []
        for idx, cls in enumerate(self.classes):
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.exists(cls_dir):
                print(f"Warning: Directory {cls_dir} does not exist!")
                continue
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                if not os.path.isfile(img_path):
                    continue
                self.data.append((img_path, idx))
        
        if len(self.data) == 0:
            raise ValueError(f"No valid images found in {root_dir}")
        
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
    dX = right_eye[0] - right_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))
    eyes_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
    M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
    aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return aligned

# Face Classifier - Abhinav Model
# class FaceClassifier(nn.Module):
#     def __init__(self, input_dim=512, num_classes=5):
#         super(FaceClassifier, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(input_dim, 256),
#             nn.ReLU(),
#             nn.Dropout(0.4),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Dropout(0.4),
#             nn.Linear(128, num_classes),
#             nn.Softmax(dim=1)
#         )
    
#     def forward(self, x):
#         return self.fc(x)

# Face Classifier - Adarsh Model
class FaceClassifier(nn.Module):
    def __init__(self, input_dim=512, num_classes=4):  # 3 people + imposter
        super(FaceClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
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

# NASDistortionNet
class MixedOp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.ops = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 5, padding=2), nn.ReLU()),
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=2, dilation=2), nn.ReLU()),
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.Dropout2d(0.3), nn.ReLU())
        ])
        self.alpha = nn.Parameter(torch.randn(len(self.ops)))

    def forward(self, x):
        weights = torch.nn.functional.softmax(self.alpha, dim=0)
        return sum(w * op(x) for w, op in zip(weights, self.ops))

class NASDistortionNet(nn.Module):
    def __init__(self, noise_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout2d(0.2),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.Dropout2d(0.2)
        )
        self.mixed1 = MixedOp(128, 128)
        self.mixed2 = MixedOp(128, 128)
        self.noise_proj = nn.Linear(noise_dim, 128 * 32 * 32)
        self.noise_norm = nn.BatchNorm2d(128)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 3, 7, padding=3), nn.Tanh(),
            nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False)
        )
        self.non_invertible_layer = nn.Sequential(
            nn.MaxPool2d(2), nn.Dropout2d(0.5), nn.GELU()
        )

    def forward(self, x, z):
        x = self.encoder(x)
        x = self.mixed1(x)
        x = self.mixed2(x)
        B, _, H, W = x.shape
        z_proj = self.noise_proj(z).view(B, 128, H, W)
        z_proj = self.noise_norm(z_proj)
        x = torch.cat([x, z_proj], dim=1)
        x = self.non_invertible_layer(x)
        return self.decoder(x)

# Secure seed and noise vector generation
def secure_seed(user_id: str, revocation_key: str) -> int:
    key_bytes = revocation_key.encode('utf-8')
    msg_bytes = user_id.encode('utf-8')
    hmac_digest = hmac.new(key_bytes, msg_bytes, hashlib.sha256).hexdigest()
    return int(hmac_digest, 16) % (2**32)

def generate_z(user_id, revocation_key, noise_dim=128):
    seed = secure_seed(user_id, revocation_key)
    generator = torch.Generator().manual_seed(seed)
    return torch.randn((1, noise_dim), generator=generator).squeeze(0)

# Load NASDistortionNet model
distortion_model = NASDistortionNet().to(device)
distortion_model_path = "trained_models/nas_distortion.pt"
state_dict = torch.load(distortion_model_path, map_location=device)
distortion_model.load_state_dict(state_dict)
distortion_model.eval()

# Preprocessing for distortion
distortion_preprocess = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Process frame with improved bounding box handling
last_detection = None
SMOOTHING_FRAMES = 2    
frame_count_since_detection = 0

# Process frame with improved detection and no caching
def process_frame(frame, classes, confidence_threshold=0.7):
    # Log frame processing start
    print("Processing new frame")
    
    # Preprocess frame: histogram equalization for better detection
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_yuv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2YUV)
    frame_yuv[:, :, 0] = cv2.equalizeHist(frame_yuv[:, :, 0])
    frame_rgb = cv2.cvtColor(frame_yuv, cv2.COLOR_YUV2RGB)
    
    # Try primary MTCNN detection
    boxes, _, landmarks = mtcnn.detect(frame_rgb, landmarks=True)
    print(f"Primary MTCNN detection: {'Success' if boxes is not None and len(boxes) > 0 else 'Failed'}")
    
    # Retry with more lenient parameters if detection fails
    if boxes is None or len(boxes) == 0:
        mtcnn_temp = MTCNN(
            keep_all=True,
            device=device,
            min_face_size=15,
            thresholds=[0.4, 0.5, 0.6]
        )
        boxes, _, landmarks = mtcnn_temp.detect(frame_rgb, landmarks=True)
        print(f"Fallback MTCNN detection: {'Success' if boxes is not None and len(boxes) > 0 else 'Failed'}")
    
    if boxes is not None and len(boxes) > 0:
        # Process the first detected face
        x, y, x2, y2 = boxes[0]
        x, y, x2, y2 = int(max(0, x)), int(max(0, y)), int(min(frame.shape[1], x2)), int(min(frame.shape[0], y2))
        width = x2 - x
        height = y2 - y
        box_size = (width, height)
        
        print(f"Detected box: x={x}, y={y}, x2={x2}, y2={y2}, width={width}, height={height}")
        
        # Extract face region
        face = frame_rgb[y:y2, x:x2]
        if face.size == 0 or width <= 0 or height <= 0:
            print("Invalid face region (empty or negative dimensions)")
            return frame, None, None, None
        
        try:
            face = cv2.resize(face, (160, 160))
            # face = align_face(face, landmarks[0])
            face_tensor = torch.tensor(face).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            print("Face tensor created successfully")
        except Exception as e:
            print(f"Error processing face: {e}")
            return frame, None, None, None
        
        # Get embedding and classify
        with torch.no_grad():
            embedding = resnet(face_tensor)
            output = classifier(embedding)
        probs = output.detach().cpu().numpy()[0]
        max_confidence = np.max(probs)
        identity = classes[np.argmax(probs)]
        
        print(f"Classification: identity={identity}, confidence={max_confidence * 100:.2f}%")
        
        # Apply confidence threshold
        # if max_confidence < confidence_threshold:
        #     print(f"Confidence below threshold: {max_confidence * 100:.2f}%")
        #     frame = cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
        #     return frame, "imposter", max_confidence * 100, box_size
        
        # Draw bounding box and return result
        frame = cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
        return frame, identity, max_confidence * 100, box_size
    
    print("No face detected in frame")
    return frame, None, None, None

# Distort and compare function (updated to compute SSIM, PSNR, MSE)
def distort_and_compare(frame, user_id, revocation_key, classes, template_dir="/Users/adarsh.vasanthappa/Desktop/College/Capstone/final_year/cancellable_template"):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, _, landmarks = mtcnn.detect(frame_rgb, landmarks=True)
    
    if boxes is None or len(boxes) == 0:
        print("No face detected in frame")
        return None, None, None, None, None
    
    x, y, x2, y2 = boxes[0]
    x, y, x2, y2 = int(max(0, x)), int(max(0, y)), int(min(frame.shape[1], x2)), int(min(frame.shape[0], y2))
    face = frame_rgb[y:y2, x:x2]
    
    if face.size == 0 or x2 <= x or y2 <= y:
        print("Invalid face region")
        return None, None, None, None, None
    
    # Resize and preprocess input face (no distortion)
    face = cv2.resize(face, (128, 128))
    face_pil = Image.fromarray(face)
    face_tensor = distortion_preprocess(face_pil).unsqueeze(0).to(device)
    face_tensor = (face_tensor + 1) / 2  # Unnormalize to [0, 1]
    
    best_identity = None
    best_ssim = 0.0
    best_psnr = 0.0
    best_mse = float('inf')
    ssim_threshold = 0.2
    
    for cls in classes:
        template_path = os.path.join(template_dir, cls, "distorted_face.png")
        print(f"Processing template: {template_path}")
        if not os.path.exists(template_path):
            print(f"Template not found: {template_path}")
            continue
        template_img = Image.open(template_path).convert("RGB")
        template_tensor = distortion_preprocess(template_img).unsqueeze(0).to(device)
        template_tensor = (template_tensor + 1) / 2  # Unnormalize to [0, 1]
        
        # Compute metrics
        ssim_value = ssim(face_tensor, template_tensor, data_range=1.0).item()
        psnr_value = psnr(face_tensor, template_tensor, data_range=1.0).item()
        mse_value = torch.mean((face_tensor - template_tensor) ** 2).item()
        
        print(f"Metrics for {cls}: SSIM={ssim_value:.4f}, PSNR={psnr_value:.4f}, MSE={mse_value:.4f}")
        
        if ssim_value > best_ssim:
            best_ssim = ssim_value
            best_psnr = psnr_value
            best_mse = mse_value
            best_identity = cls
    
    if best_ssim >= ssim_threshold:
        print(f"Match found: {best_identity} with SSIM={best_ssim:.4f}, PSNR={best_psnr:.4f}, MSE={best_mse:.4f}")
        return face_tensor, best_identity, best_ssim, best_psnr, best_mse
    print(f"No match: best SSIM={best_ssim:.4f} below threshold")
    return face_tensor, "imposters", best_ssim, best_psnr, best_mse

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
            
            loss_ce = ce_loss(outputs, labels)
            anchor = embeddings[0:1]
            positive = embeddings[1:2] if labels[1] == labels[0] else embeddings[2:3]
            negative = embeddings[2:3] if labels[2] != labels[0] else embeddings[1:2]
            loss_triplet = triplet_loss(anchor, positive, negative)
            
            loss = loss_ce + 0.5 * loss_triplet
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
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
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'resnet_state_dict': resnet.state_dict(),
                'classifier_state_dict': classifier.state_dict()
            }, model_path)

def classify_person(image, classes=['abhinav', 'adarsh', 'arshini', 'imposters'], confidence_threshold=0.7):
    if image is None:
        return None, None
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.convertScaleAbs(img_rgb, alpha=1.1, beta=5)
    
    boxes, _, landmarks = mtcnn.detect(img_rgb, landmarks=True)
    
    if boxes is None or len(boxes) == 0:
        mtcnn_temp = MTCNN(keep_all=True, device=device, min_face_size=20, thresholds=[0.5, 0.6, 0.7])
        boxes, _, landmarks = mtcnn_temp.detect(img_rgb, landmarks=True)
    
    if boxes is None or len(boxes) == 0:
        print("No face detected in the image")
        return None, None
    
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
    
    x, y, x2, y2 = int(max(0, x)), int(max(0, y)), int(min(img_rgb.shape[1], x2)), int(min(img_rgb.shape[0], y2))
    if x2 <= x or y2 <= y:
        print("Invalid face bounding box")
        return None, None
    
    face = img_rgb[int(y):int(y2), int(x):int(x2)]
    if face.size == 0:
        print("Empty face region extracted")
        return None, None
    
    face = cv2.resize(face, (160, 160))
    face = align_face(face, landmarks[0])
    face_tensor = torch.tensor(face).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    
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