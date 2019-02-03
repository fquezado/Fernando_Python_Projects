import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as tran
import torch.utils.data as tud


batch = 2
epoch_runs = 5

# making training dataset and testing dataset
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=tran.ToTensor(), download=True)
testset = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=tran.ToTensor(), download=False)

# loading training dataset and testing dataset
train_dataset_loader = tud.DataLoader(dataset=trainset, batch_size=batch, shuffle=True)
test_dataset_loader = tud.DataLoader(dataset=testset, batch_size=batch, shuffle=False)


# MODEL
class FashionMNISTCNN(nn.Module):
    def __init__(self):
        super(FashionMNISTCNN, self).__init__()
        # 1 channel, images are grey scale
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1)
        self.relu1 = nn.ReLU()

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1)
        self.relu2 = nn.ReLU()

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.fullyC = nn.Linear(32*4*4, 10)  # why is it 4 * 4, I understand the 32 part

        # technically 4 layers - convolution1, max-pooling, convolution2, fully connected
    def forward(self, input):
        out = self.conv1(input)
        out = self.relu1(out)

        out = self.pool1(out)

        out = self.conv2(out)
        out = self.relu2(out)

        out = self.pool2(out)

        # print(out.shape) important to find right shape for flattening!

        out = out.view(out.size(0), -1)  # flattening in order to use Linear fully connected layer
        out = self.fullyC(out)

        return out


# making optimizer function and loss function


model = FashionMNISTCNN()
loss_function = nn.CrossEntropyLoss()  # criterion, otherwise called loss function
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # initiating optimizer function


# TRAINING
for epoch in range(epoch_runs):
    running_loss = 0.0
    for i, x in enumerate(train_dataset_loader): # CHECKKKKKKKKKKKKKKKKK
        CNN_train_inputs, CNN_train_labels = x  # getting in inputs from for-loop

        optimizer.zero_grad()  # zero's out the gradients of the parameters

        CNN_outputs = model(CNN_train_inputs)  # forward
        loss_function(CNN_outputs, CNN_train_labels).backward()  # backward
        optimizer.step()  # optimize


# ACCURACY OF TRAINED MODEL

total_test_set = 0  # about 10,000 images
correct_answers = 0 # used to get generalization
with torch.no_grad():  # confused by this line, what does it do to the gradients
    for data in test_dataset_loader:  # for loop going through
        CNN_test_inputs, CNN_test_labels = data  # getting in inputs from for-loop for test data
        CNN_outputs = model(CNN_test_inputs)  # forward pass?
        _, predict = torch.max(CNN_outputs.data, 1)  # ? what does "_," do???!!
        total_test_set = total_test_set + CNN_test_labels.size(0)  # gets number of labels in test set/# of images
        correct_answers = correct_answers + (predict == CNN_test_labels).sum().item()  # review what this means??

print("Accuracy of Fashion MNIST dataset equals " + str(100 * correct_answers/total_test_set))
# Getting about 86 percent accuracy with current settings, layers, and etc...
        
