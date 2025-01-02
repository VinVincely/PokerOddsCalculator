# Imports 
import torch, os
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from PIL import Image


def get_classLabels(dataDir):
    ''' Get Class Labels '''
    classLabels = {}; idx = 0
    fileFolders = os.listdir(dataDir)
    for file in fileFolders:
        filePath = dataDir + file
        if os.path.isdir(filePath):
            classLabels[idx] = file; idx += 1
    return classLabels



class PlayingCards_Dataset(Dataset):    
    def __init__(self, dataDir, transform=None):
        """
        Args: 
            dataDir (str): Root Directory with individual card folders
            transform: Optional transform to be applied on images 
        """                
        self.data = ImageFolder(dataDir, transform=transform)     
        self.transform = transform
        
        # self.classes = np.arange(0, 52, 1)

        self.classLabels = get_classLabels(dataDir)

        # List entire dataset 
        self.image_paths = []
        self.image_class = []
        for idx in self.classLabels:
            class_path = dataDir + self.classLabels[idx] + '/'
            files = os.listdir(class_path)
            for file in files: 
                self.image_paths.append(class_path + file)
                self.image_class.append(idx)
                                    
    def __len__(self):
        """ Returns the length of the dataset 
         (i.e. how many examples are in the dataset) """
        return len(self.image_paths)
    
    def __getitem__(self, index):
        """ Takes the index of one item and returns the corresponding image & label """
        imagePath = self.image_paths[index]
        image = Image.open(imagePath).convert('L')
        label = self.image_class[index]

        if self.transform:
            print(f'shape & Type of image into tranform: {image.size}, {image.mode}')
            image = self.transform(image)
        
        return image, label
    
    def __str__(self):
        ''' Print key metrics on the dataset, including number of samples and distribution across each class'''
        pass
        
# Define transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to standard size
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                     std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

def create_data_loaders(data_dir, batch_size=32, train_split=0.8, 
                       random_seed=42, num_workers=4):
    """
    Create train and validation data loaders with error handling
    """
    
    # Create dataset
    print('Loading the Playing Cards Dataset...')
    full_dataset = PlayingCards_Dataset(data_dir, transform=transform)
    print(f"Found {len(full_dataset)} images across {len(full_dataset.classLabels)} classes!")
    
    # Calculate lengths for split
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size
    print(f"Splitting the data into {train_size} training and {val_size} validation samples.\n")

    # Split dataset
    torch.manual_seed(random_seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,  # Added for better performance
        drop_last=True    # Added to ensure consistent batch sizes
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, full_dataset.classLabels


# Define the model architecture
class PlayingCard_Classifier(nn.Module):
    def __init__(self):
        super(PlayingCard_Classifier, self).__init__()

        self.conv_layers = nn.Sequential(

             # Input: 128x128x1
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # 128x128 -> 128x128
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128x128 -> 64x64
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 64x64 -> 64x64
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x64 -> 32x32
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 32x32 -> 32x32
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32 -> 16x16
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 16x16 -> 16x16
            nn.ReLU(),
            nn.MaxPool2d(2)  # 16x16 -> 8x8
        )
 
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 53)
        )

    def forward(self, x):
        # print(f'Input shape: {x.shape}')
        x = self.conv_layers(x)
        # print(f'Shape after conv layer: {x.shape}')
        x = x.view(x.size(0), -1)  # Flatten
        # print(f"Shape after flattening: {x.shape}")
        x = self.fc_layers(x)
        # print(f'Output shape: {x.shape}')
        return x


# Training method
def train_classifier(model, train_loader, num_epochs=5, learning_rate=1e-3):

    # Loss and optimizer 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training the network
    num_correct = 0; total = 0; 
    for epoch in range(num_epochs):
        runningLoss = 0
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):

            # forward 
            scores = model(data)
            loss = criterion(scores, targets)

            # backward 
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

            # Track Key metrics 
            runningLoss += loss.item()
            _, prediction = scores.max(1)
            total += targets.size(0)
            num_correct += sum(prediction == targets)

            # print(f'Running Loss: {runningLoss}')
            # print(f'Prediction: {prediction}')
            # print(f'total: {total}')
            # print(f'num_correct: {num_correct}')

            # print(f'Target Size: {targets.shape}')
            # print(f'Prediciton Size: {prediction.shape}')
                        
        epoch_loss = runningLoss / len(train_loader)
        epoch_accuracy = num_correct / total

        print(f'\nEpoch #{epoch+1}/{num_epochs}: Model Loss = {epoch_loss} & Accuracy = {epoch_accuracy}\n')


def evaluate_classifier(model, test_loader):
    num_correct = 0; total = 0; 
    for batch_idx, (data, targets) in enumerate((test_loader)):
        
        scores = model(data)

        # Track Key metrics         
        _, prediction = scores.max(1)
        total += targets.size(0)
        num_correct += sum(prediction == targets)
    
    print(f'Accuracy = {num_correct/total} ({total} samples)')


def main():
    data_dir = './Cards_Dataset/'
    train_loader, val_loader, classes = create_data_loaders(data_dir)

    model = PlayingCard_Classifier()
    
    # print('Begining Classifer Training:')
    # train_classifier(model, train_loader, num_epochs=10)

    # print('\nTesting model performance on validation dataset:')
    # evaluate_classifier(model, val_loader)
    
    # file = 'CardClassifier.pth'
    # filepath = data_dir + file
    # print('\nSaving Classifer...', end='')
    # torch.save(model, filepath)
    # print(f'Saved to {filepath}!\n')




    # # Loading Model 
    # file = 'CardClassifier.pth'
    # filepath = data_dir + file
    # model = torch.load(filepath)

    # Visualize traning data 
    batch_idx = 0; x = 0  
    plt.figure(figsize=(7,7))
    for batch_idx, (data, targets) in enumerate((val_loader)):
        
        print(data)
        print(data.shape)
        print(data.type)

        
        score = model(data)        
        _, prediction = score.max(1)

        # print(prediction)

        for img in data:
            
            # classLabel = str(classes[int(targets[x].numpy())])
            classLabel = str(classes[prediction[x].item()])


            plt.subplot(3,3,x+1); plt.imshow(img[0]); 
            plt.axis('off'); 
            plt.text(10,20,classLabel,fontsize=16,color='white')
            if x == 8: break
            x += 1
        break 
    plt.tight_layout()
    plt.show()
    
    
    # # Test Model 
    # model = PlayingCard_Classifier()
    # print('Model Architecture:'); print(model)
    # x = torch.rand((1,1,128,128)); #print(x.shape)
    # # images, labels = enumerate(train_loader)
    # out = model(x)
    # print(out)

    # Test Training 
    

if __name__ == "__main__":
    # Example usage with default parameters (10-second recording)
    main()
