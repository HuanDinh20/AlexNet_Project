import utils
import train
from module import AlexNet
import torch
from torch import nn
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# 1. define some args
root = r'A:\huan_shit\Study_Shit\Deep_Learning\Side_Projects\LeNet_project\data'
batch_size = 32
EPOCH = 15
device = utils.get_device()
summary_writer = SummaryWriter()
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_path = f'save_model\pth_model\AlexNet'

# 2, prepare dataset
aug_train, aug_test = utils.image_augmentation()
train_set, validation_set = utils.get_FashionMNIST(root, aug_train, aug_test)
train_dataloader, validation_dataloader = utils.create_dataloader(train_set, validation_set, batch_size=batch_size)

# 3. create loss function

loss_fn = nn.CrossEntropyLoss()

# 4. call model and move it to device
model = AlexNet()
model.to(device)

# 5. create optimizer

optimizer = torch.optim.Adam(model.parameters())

if __name__ == '__main__':
    train.per_epoch_activity(train_dataloader, validation_dataloader, device,optimizer, model, loss_fn,summary_writer,
                             timestamp, epochs=EPOCH)
    print(" training complete")
    print('*'*10+'Saving model'*10)
    model_path = model_path + str(timestamp) + '.pth'
    torch.save(model, model_path)
    print('model is saved at: ', model_path + str(timestamp) + '.pth')


