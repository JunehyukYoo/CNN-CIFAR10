import torch 
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def main():
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Set the batch size for loading the dataset
    # The batch size determines the number of samples that will be propagated through the network at once.
    # A larger batch size can lead to faster training, but requires more memory.
    # A smaller batch size may lead to more stable training, but can be slower.
    # In this case, we set the batch size to 128, which is a common choice for training deep learning models.
    batch_size = 128

    # Download and load the CIFAR-10 dataset
    # The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class.
    # The dataset is divided into 50,000 training images and 10,000 test images.
    # The training set is used to train the model, while the test set is used to evaluate its performance.
    # The dataset is downloaded from the internet and stored in the specified root directory.
    # The transform applied to the images includes converting them to tensors and normalizing them.
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Visualizing some of the images
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.savefig("./examples/randomtrainingimages.png", bbox_inches="tight")


    # Get random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # Show images (saved to ./randomtrainingimages.png)
    imshow(torchvision.utils.make_grid(images))

    # Define a Convolutional Neural Network
    # The network consists of two convolutional layers followed by three fully connected layers.
    # The convolutional layers use ReLU activation and max pooling, while the fully connected layers also use ReLU activation.
    # The final layer outputs 10 classes corresponding to the CIFAR-10 dataset.
    # The network is defined as a subclass of nn.Module, and the forward method specifies the forward pass through the network.
    # The forward method applies the convolutional layers, activation functions, and pooling operations in sequence.
    # The input to the network is a batch of images, and the output is the predicted class scores for each image.
    # The network is designed to take 3-channel images (RGB) as input and output 10 class scores.
    # class Net(nn.Module):
    #     def __init__(self):
    #         super().__init__()
    #         self.conv1 = nn.Conv2d(3, 6, 5)
    #         self.pool = nn.MaxPool2d(2, 2)
    #         self.conv2 = nn.Conv2d(6, 16, 5)
    #         self.fc1 = nn.Linear(16 * 5 * 5, 120)
    #         self.fc2 = nn.Linear(120, 84)
    #         self.fc3 = nn.Linear(84, 10)

    #     def forward(self, x):
    #         x = self.pool(F.relu(self.conv1(x)))
    #         x = self.pool(F.relu(self.conv2(x)))
    #         x = torch.flatten(x, 1) # flatten all dimensions except batch
    #         x = F.relu(self.fc1(x))
    #         x = F.relu(self.fc2(x))
    #         x = self.fc3(x)
    #         return x

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Conv2d(3,16,8,1,1),
                nn.ReLU(),
                nn.MaxPool2d(2,2),
                nn.Conv2d(16,32,8,1,1),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.Dropout(p=0.2),
                nn.Flatten(), 
                nn.Linear(2048, 64),
                nn.ReLU(),
                nn.Linear(64, 10)

            )
        def forward(self, x):
            return self.network(x)

    # Create an instance of the network
    net = Net()
    # Move the network to the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    
    # Define a loss function and optimizer
    # The loss function is CrossEntropyLoss, which is commonly used for multi-class classification problems.
    # The optimizer is SGD (Stochastic Gradient Descent) with a learning rate (alpha) of 0.001 and momentum of 0.9.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Train the network
    # The training loop iterates over the training data for a specified number of epochs.
    # In each iteration, the inputs and labels are obtained from the data loader.
    # The optimizer's gradients are zeroed, and the forward pass is performed to compute the outputs.
    # The loss is computed using the criterion, and the backward pass is performed to compute the gradients.
    # The optimizer updates the network parameters based on the computed gradients.
    def train(n):
        for epoch in range(1, n+1):  
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                # Move the inputs and labels to the GPU if available
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0

            print(f'Epoch {epoch} finished.')

    # Train the network for 5 epochs
    train(30)

    # Save the trained model
    def save_model():
        PATH = './cifar_net.pth'
        torch.save(net.state_dict(), PATH)
        print(f'Model saved to {PATH}')

    # Load the trained model
    def load_model():
        PATH = './cifar_net.pth'
        net.load_state_dict(torch.load(PATH))
        print(f'Model loaded from {PATH}')

    # Print the accuracy of the network on the test data
    def total_accuracy():
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                # calculate outputs by running images through the network
                outputs = net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    # Print the accuracy of the network on the test data
    def accuracy_per_class():
        # prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                # calculate outputs by running images through the network
                outputs = net(images)
                # the class with the highest energy is what we choose as prediction
                _, predictions = torch.max(outputs, 1)

                # collect the correct predictions for each class
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1

        # print accuracy for each class along with the mean accuracy
        mean_accuracy = 0
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
            mean_accuracy += accuracy

        print(f'Mean Accuracy: {mean_accuracy / len(classes):.1f} %')
    
    total_accuracy()
    accuracy_per_class()

if __name__ == "__main__":
    main()