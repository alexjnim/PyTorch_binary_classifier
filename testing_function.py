import torch
from resources.Dataset import generate_batches
from config import config

def test_model(model, train_state, device, dataset, loss_func, compute_accuracy):
    model.load_state_dict(torch.load(config.save_dir+config.model_filename))
    model = model.to(device)

    dataset.set_split('test')
    batch_generator = generate_batches(dataset,
                                    batch_size=config.batch_size,
                                    device=device)
    running_loss = 0.
    running_acc = 0.
    model.eval()

    for batch_index, batch_dict in enumerate(batch_generator):
        print('test batch')
        # compute the output
        y_pred = model(x_in=batch_dict['x_data'].float())

        # compute the loss
        loss = loss_func(y_pred, batch_dict['y_target'].float())
        loss_t = loss.item()
        running_loss += (loss_t - running_loss) / (batch_index + 1)

        # compute the accuracy
        acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
        running_acc += (acc_t - running_acc) / (batch_index + 1)

    train_state['test_loss'] = running_loss
    train_state['test_acc'] = running_acc

    return train_state