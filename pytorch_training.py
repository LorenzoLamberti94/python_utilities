import torch
import torch.nn as nn
from tqdm import tqdm

from sklearn.metrics import accuracy_score


def training(net, train_loader, val_loader, loss_function, epochs):   
    """
    Training loop for a PyTorch model.
    Args:
        net: PyTorch model
        train_loader: PyTorch DataLoader for training data
        val_loader: PyTorch DataLoader for validation data
        loss_function: PyTorch loss function
        epochs: number of epochs to train
    """ 
    
    # count tot batches
    tot_train_batches = len(train_loader)
    tot_val_batches = len(val_loader)

    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)  # Define optimizer

    for epoch in range(epochs): # iterate over epochs

        # initialize variables to store training loss and accuracy
        total_epoch_loss = 0
        train_accuracy = []
        
        # initialize progress bar
        progress = tqdm( 
            enumerate(train_loader), 
            desc="Loss: ", 
            total=tot_train_batches) 

        # ============================= TRAINING ============================= #
        
        net.train() # set model to training mode
    
        for batch_idx, data in progress: # iterate over batches
            inputs, labels = data[0].to(device), data[1].to(device) # get inputs and labels and move data to device
            outputs = net(inputs) # forward pass 
            
            loss = loss_function(outputs, labels) # compute loss
            net.zero_grad() # clear previous gradients
            loss.backward() # compute gradients
            current_loss = loss.item() # get current loss
            optimizer.step() # update weights

            # calculate training accuracy
            predicted_classes = torch.max(outputs, 1)[1] # get class from network's prediction
            train_accuracy.append(calculate_accuracy(predicted_classes.cpu(), labels.cpu())) 
            average_train_accuracy = sum(train_accuracy)/(batch_idx+1)

            #compute average loss
            total_epoch_loss += current_loss
            average_train_loss = total_epoch_loss/(batch_idx+1)
            
            # updating progress bar
            progress.set_description("Training Loss: {:.4f}, Accuracy: {:.4f}".format(average_train_loss, average_train_accuracy))
            

        # ============================ VALIDATION ============================ #
        val_accuracy, average_val_loss = validate(net, 
                                                  val_loader, 
                                                  loss_function, 
                                                  accuracy_score
                                                  )
        
        # compute mean accuracy
        average_train_accuracy = sum(train_accuracy)/tot_train_batches
        average_val_accuracy = sum(val_accuracy)/tot_val_batches

        # print training/validation Accuracy and Loss
        print('Epoch %d/%d' % (epoch+1,epochs), 'Training loss:  %.4f' % (average_train_loss), 'Accuracy: %.4f' % (average_train_accuracy))
        print('Validation Loss: %.4f' % (average_val_loss), 'Accuracy: %.4f' % (average_val_accuracy))


def get_data_loaders(train_batch_size=128, val_batch_size=128, path='./data/fashionmnist',verbose=False):
    # Load FashionMNIST dataset 
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    
    #define data transformations
    data_transform = transforms.Compose([transforms.ToTensor()])
    
    # Load train/valid datasets for FashionMNIST
    train_dataset = datasets.FashionMNIST(download=True, 
                                          root=path, 
                                          transform=data_transform, 
                                          train=True
                                          
                                          )
    valid_dataset = datasets.FashionMNIST(download=False, 
                                          root=path, 
                                          transform=data_transform, 
                                          train=False
                                          )
    
    # define dataloaders
    train_loader = torch.utils.data.DataLoader( train_dataset, 
                                                batch_size=train_batch_size, 
                                                shuffle=True
                                                )
    
    valid_loader = torch.utils.data.DataLoader( valid_dataset, 
                                                batch_size=val_batch_size, 
                                                shuffle=False
                                                )

    # count how many images we have in each set
    train_dataset_size = len(train_dataset)
    valid_dataset_size = len(valid_dataset)
    if verbose: print('Dataset size:\nTraining images %d,\nValidation images %d' % (train_dataset_size, valid_dataset_size))
    
    return train_loader, valid_loader


        
def main():
    global device # define device as global variable
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # define device

    net = CNN() # define net

    net.to(device) # Move net to gpu/cpu

    train_loader, 
    val_loader = get_data_loaders(batch_size, 
                                  batch_size) # Build Dataloaders

    loss_function = nn.CrossEntropyLoss() # CrossEntropyLoss for classification
    
    training(net, train_loader, 
             val_loader, 
             loss_function, 
             epochs=100
             )
      
if __name__ == '__main__':
    main()
