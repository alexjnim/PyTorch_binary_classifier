import json
import torch
from resources.Dataset import generate_batches
from testing_function import test_model
from config import config

def get_train_state():
    return  {'epoch_index': 0,
                'train_loss': [],
                'train_acc': [],
                'val_loss': [],
                'val_acc': [],
                'test_loss': -1,
                'test_acc': -1,
                'model_filename': config.model_filename}

def set_seed_everywhere(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)

def compute_accuracy(y_pred, y_target):
    # .cpu() is used to move the tensor to cpu(). Some operations on tensors cannot be performed on cuda tensors so you need to move them to cpu first.
    y_target = y_target.cpu()
    # use .sigmoid activation function to get probablity
    y_pred_indices = (torch.sigmoid(y_pred)>0.5).cpu().long()#.max(dim=1)[1]
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100

def save_model(train_state, model):
    # Save one model at least
    if train_state['epoch_index'] == 0:
        torch.save(model.state_dict(), config.save_dir+config.model_filename)
        train_state['stop_early'] = False

    # Save model if performance improved
    elif train_state['epoch_index'] >= 1:
        loss_tm1, loss_t = train_state['val_loss'][-2:]

        # If loss worsened
        # if loss_t <= train_state['early_stopping_best_val']:
        #    # Update step
        #     train_state['early_stopping_step'] += 1
        # # Loss decreased
        # else:
        # Save the best model
        # if loss_t < train_state['early_stopping_best_val']:
        if loss_t < loss_tm1:
            torch.save(model.state_dict(), config.save_dir+train_state['model_filename'])

            # # Reset early stopping step
            # train_state['early_stopping_step'] = 0

def train_val_model(model, dataset, device,  optimizer, loss_func):
    train_state = get_train_state()
    for epoch_index in range(config.num_epochs):
        train_state['epoch_index'] = epoch_index

        # Iterate over training dataset
        print('Iterate over training dataset')
        # setup: batch generator, set loss and acc to 0, set train mode on
        dataset.set_split('train')
        batch_generator = generate_batches(dataset,
                                        batch_size=config.batch_size,
                                        device=device)
        running_loss = 0.0
        running_acc = 0.0
        model.train()

        for batch_index, batch_dict in enumerate(batch_generator):
            # the training routine is 5 steps:
            print('train batch')
            # step 1. zero the gradients
            optimizer.zero_grad()

            # step 2. compute the output
            y_pred = model(x_in=batch_dict['x_data'].float())

            # step 3. compute the loss
            loss = loss_func(y_pred, batch_dict['y_target'].float())
            loss_batch = loss.item()
            running_loss += (loss_batch - running_loss) / (batch_index + 1)

            # step 4. use loss to produce gradients
            loss.backward()

            # step 5. use optimizer to take gradient step
            optimizer.step()

            # -----------------------------------------
            # compute the accuracy
            acc_batch = compute_accuracy(y_pred, batch_dict['y_target'])
            running_acc += (acc_batch - running_acc) / (batch_index + 1)

        train_state['train_loss'].append(running_loss)
        train_state['train_acc'].append(running_acc)

        # Iterate over val dataset
        print('Iterate over val dataset')
        # setup: batch generator, set loss and acc to 0, set eval mode on
        dataset.set_split('val')
        batch_generator = generate_batches(dataset,
                                        batch_size=config.batch_size,
                                        device=device)
        running_loss = 0.
        running_acc = 0.
        #.eval() method makes the model parameters immutable and disables dropout. also disables computation of the loss and propagation of gradients back to the parameters.
        model.eval()

        for batch_index, batch_dict in enumerate(batch_generator):
            print('val batch')
            # step 1. compute the output
            y_pred = model(x_in=batch_dict['x_data'].float())

            # step 2. compute the loss
            loss = loss_func(y_pred, batch_dict['y_target'].float())
            loss_batch = loss.item()
            running_loss += (loss_batch - running_loss) / (batch_index + 1)

            # step 3. compute the accuracy
            acc_batch = compute_accuracy(y_pred, batch_dict['y_target'])
            running_acc += (acc_batch - running_acc) / (batch_index + 1)

        train_state['val_loss'].append(running_loss)
        train_state['val_acc'].append(running_acc)
        save_model(train_state, model)

    #after finishing train and val loop, check against test data and save results
    train_state = test_model(model, train_state, device, dataset, loss_func, compute_accuracy)
    with open('results/training_results.json', 'w') as fp:
        json.dump(train_state, fp)

    return train_state, model, optimizer