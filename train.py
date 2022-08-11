import torch
def train_one_epoch(train_dataloader, device, optimizer, model, loss_fn, summary_writer, epoch_idx):
    """
    1. get a batch of data from data loader & move to device
    2. zeros the optimizer's gradients
    3. perform inference - get predcitions from model for an input batch
    4. compute loss (predict, actual)
    5. calculate backward gradients over the learning weights
    6. tell optimizer to perform learning step - adjust the learning weights based on the observed gradients, according
    to the optimize algorithm
    7. Report loss every 1000 batch
    8. Report the avg per batch loss to compare for a comparison with validation run
    """
    running_loss = 0.0
    last_loss = 0.0
    for i, data in enumerate(train_dataloader):
        # 1. get a batch of data from data loader
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # 2. Zeros the optimizer
        optimizer.zero_grad()

        # 4. perform inference
        outputs = model(inputs)

        # 5. Calculate loss
        loss = loss_fn(outputs, labels)

        # 6. Tell optimizer to perform learning step
        optimizer.step()

        # 7. Gather data and report every 1000 batches
        running_loss += loss.item()

        if bool(i % 1000):
            last_loss = running_loss / 1000
            print(f"Batch {i} loss {last_loss}")
            tb_x = epoch_idx * len(train_dataloader) + i +1
            summary_writer.add_scalar("Loss/ Train", last_loss, tb_x)
            running_loss = 0

    return last_loss


def per_epoch_activity(train_dataloader, val_dataloader, device, optimizer, model, loss_fn, summary_writer,timestamp ,epochs=2):
    """
    at each epoch:
    1. per form validation by checking relative loss on a set of data that was not use for training
        1.1 train one epoch to get the avg_loss per epoch
        1.2 untrain model, and make validation:
            a. get a batch of dataset on validation loader
            b. perfrom inference, get the val_loss and running_val_loss from
            c. get the avg_val_loss
            d. report to summary_writer

    2. save a copy of model
    """
    best_loss = 1_000_000.
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}')
        # 1.1 Train one epoch
        model.train(True)
        avg_loss = train_one_epoch(train_dataloader, device, optimizer, model, loss_fn, summary_writer, epoch)

        # 1.2 now un train it the model, to perform validation
        model.train(False)
        val_running_loss = 0.0

        for i, data in enumerate(val_dataloader):
            # a. get validation input, output
            val_inputs, val_labels = data
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

            # b. perform inference

            val_outputs = model(val_inputs)

            val_loss = loss_fn(val_outputs, val_labels)
            val_running_loss += val_loss
        # c. get avg val running loss
        avg_val_loss = val_running_loss / (i + 1)
        print(f"Training loss: {avg_loss}, Validation loss: {avg_val_loss}")

        # d. report to summary_writer
        summary_writer.add_scalar("Training Loss vs Validation Loss",
                                  {'Training Loss': avg_loss, 'Validation Loss': avg_val_loss},
                                  epoch)
        summary_writer.flush()

        # 2. Save the best performance model's state
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            model_path = f'saved_model\model_{epoch}_{timestamp}'
            torch.save(model.state_dict(), model_path)













