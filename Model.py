import torch.nn as nn

#Torch CNN model with 3 Conv layers and 3 fully connected layers
class Simple_CNN_Net(nn.Module):
    def __init__(self, num_classes=1, num_concepts=1, image_size=(192, 128)):
        super(Simple_CNN_Net, self).__init__()
        #Size reduction factor of image by pooling layers
        mod = 1

        #conv Layers
        in_channels, out_channels = (3, 16)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4, padding=0)
        mod *= 4

        in_channels, out_channels = (out_channels, 2*out_channels)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        mod *= 2

        in_channels, out_channels = (out_channels, out_channels)
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        mod *= 2

        # Fully connected layers
        self.first_linear_layer_size = out_channels * (image_size[0]//mod * image_size[1]//mod)
        self.fc1 = nn.Linear(self.first_linear_layer_size, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc_concepts = nn.Linear(64, num_concepts)
        # Test additional Linear Layer: self.fc_3 = nn.Linear(num_concepts, num_concepts)
        self.fc_outputs = nn.Linear(num_concepts, num_classes)


        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Convolutional layers with activation and pooling
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = self.pool3(x)

        # Flatten the output for fully connected layers
        x = x.view(-1, self.first_linear_layer_size)  # Corrected input size based on spatial dimensions

        # Fully connected layers with activation
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc_concepts(x))
        x_concepts = x
        # Test additional Linear Layer: x = self.relu(self.fc_3(x))
        x_outputs = self.softmax(self.fc_outputs(x))

        return x_concepts, x_outputs




# Comparison model where true concepts are passed onto last layer, instead of predicted concepts.
class Simple_CNN_PerfectConcepts(nn.Module):
    def __init__(self, num_classes=1, num_concepts=1, image_size=(192, 128)):
        super(Simple_CNN_PerfectConcepts, self).__init__()
        #Size reduction factor of image by pooling layers
        mod = 1

        #conv Layers
        in_channels, out_channels = (3, 16)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4, padding=0)
        mod *= 4

        in_channels, out_channels = (out_channels, 2*out_channels)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        mod *= 2

        in_channels, out_channels = (out_channels, out_channels)
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        mod *= 2

        # Fully connected layers
        self.first_linear_layer_size = out_channels * (image_size[0]//mod * image_size[1]//mod)
        self.fc1 = nn.Linear(self.first_linear_layer_size, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc_concepts = nn.Linear(64, num_concepts)
        self.fc_outputs = nn.Linear(num_concepts, num_classes)


        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, perfectConcepts):
        # Convolutional layers with activation and pooling
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = self.pool3(x)

        # Flatten the output for fully connected layers
        x = x.view(-1, self.first_linear_layer_size)  # Corrected input size based on spatial dimensions

        # Fully connected layers with activation
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x_concepts = self.sigmoid(self.fc_concepts(x))
        x_outputs = self.softmax(self.fc_outputs(perfectConcepts))

        return x_concepts, x_outputs

# Model where only the output label is predicted and not the concepts
class Concept_To_Label_Net(nn.Module):
    def __init__(self, num_classes=1, num_concepts=1, image_size=(192, 128)):
        super(Concept_To_Label_Net, self).__init__()
        self.linear = nn.Linear(num_concepts, num_concepts)
        self.fc_outputs = nn.Linear(num_concepts, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, perfectConcepts):
        x_concepts = perfectConcepts
        x = self.relu(self.linear(x_concepts))
        x_outputs = self.softmax(self.fc_outputs(x))

        return x_concepts, x_outputs