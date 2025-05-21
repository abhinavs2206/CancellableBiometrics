from backend import FaceDataset, train_model, data_transforms

# Path to your dataset
DATASET_PATH = r"/Users/adarsh.vasanthappa/Desktop/College/Capstone/final_year/enrolled_subjects"  # Update this path
MODEL_PATH = r"/Users/adarsh.vasanthappa/Desktop/College/Capstone/final_year/trained_models/model_mac_1.pth"

# Initialize dataset with augmentation
dataset = FaceDataset(DATASET_PATH, transform=data_transforms)

# Train the model
train_model(dataset, MODEL_PATH, epochs=50)