from backend import FaceDataset, train_model, data_transforms

# Path to your dataset
DATASET_PATH = r"C:\Users\Abhinav Somisetty\Final_Project\enrolled_subjects"  # Update this path
MODEL_PATH = r"C:\Users\Abhinav Somisetty\Final_Project\model.pth"

# Initialize dataset with augmentation
dataset = FaceDataset(DATASET_PATH, transform=data_transforms)

# Train the model
train_model(dataset, MODEL_PATH, epochs=50)