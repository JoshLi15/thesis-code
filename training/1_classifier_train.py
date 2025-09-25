import argparse
import configparser
import os
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import matplotlib.pyplot as plt

from data_io.SpeakerDataset import SpeakerDataset
from models.losses import ClassBalancedLoss
from models.classifiers import PureClassifier
from models.early_stopping import EarlyStopping

# Note this code was modified from: 
#   https://arxiv.org/abs/2405.19796
#   title: Explainable Attribute-Based Speaker Verification
#   authors: {Xiaoliang Wu and Chau Luu and Peter Bell and Ajitha Rajan
#   year: 2024

# Sadly the in the paper referenced original code is not available anymore: 
#   https://anonymous.4open.science/r/explainable-SV-E3C2


def save_checkpoint(epoch, model, optimizer, loss, filename, is_best=False):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(os.path.dirname(filename), 'best_checkpoint.pt')
        torch.save(state, best_filename)


def main(args):
    best_val_accuracy = 0.0
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print('=' * 30)
    print('USE_CUDA SET TO: {}'.format(use_cuda),flush=True)
    print('CUDA AVAILABLE?: {}'.format(torch.cuda.is_available()),flush=True)
    print('=' * 30)
    device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")
    config_path = args.config
    config = configparser.ConfigParser()
    config.read(config_path)
    checkpoint_dir = config.get('training', 'checkpoint_dir')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)    
    

    train_data_path = config['data']['train_data']
    test_data_path = config['data']['val_data']
    classifier_heads= config['model']['classifier_heads']
    num_features = int(config['model']['num_features'])
    dropout = float(config['model']['dropout'])
    epochs = int(config['training']['epochs'])
    batch_size = int(config['training']['batch_size'])
    learning_rate = float(config['training']['learning_rate'])
    patience = int(config['training']['patience'])
   

    eval_interval = int(config.get('training', 'eval_interval'))
    log_file = os.path.join(checkpoint_dir, 'train.log')

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    ds_train = SpeakerDataset(train_data_path)
    # get the class -> int mapping for the classifier head
    print(f"Classifier head: {ds_train.get_class_encs()}", flush=True)

    num_classes=ds_train.num_classes[classifier_heads]
    class_enc_dict = ds_train.get_class_encs()
    ds_test = SpeakerDataset(test_data_path, test_mode=True, 
                                    class_enc_dict=class_enc_dict)
        
    model = PureClassifier(in_dim=num_features, out_dim=num_classes, dropout=dropout)
    model = model.to(device)

    class_counts = ds_train.get_class_counts()[classifier_heads]
    samples_per_class = list(class_counts.values())
    criterion = ClassBalancedLoss(samples_per_class, beta=0.999)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)
   
    early_stopper_acc = EarlyStopping(patience=patience, min_delta=0.001)
    early_stopper_loss = EarlyStopping(patience=patience, min_delta=0.0)

    with open(log_file, 'a') as log:
        for epoch in range(epochs):
            model.train()
            train_loss, correct_train, total_train = 0, 0, 0

            for train_inputs, train_label_all in ds_train.get_batches(batch_size=batch_size):
                train_inputs = train_inputs.to(device)
                train_labels = train_label_all[classifier_heads].to(device)

                optimizer.zero_grad()
                outputs = model(train_inputs)
                loss = criterion(outputs, train_labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct_train += (preds == train_labels).sum().item()
                total_train += train_inputs.size(0)

            train_accuracy = 100 * correct_train / total_train
            log.write(f"Epoch: {epoch + 1}, Train Loss: {train_loss}, Train Acc: {train_accuracy}\n")
            print(f"Epoch: {epoch + 1}, Train Loss: {train_loss}, Train Acc: {train_accuracy}",flush=True)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            # Evaluate Epoch with val_data set
            if (epoch + 1) % eval_interval == 0:
                val_loss, correct_val = 0, 0
                with torch.no_grad():
                    model.eval()
                    test_feats, label_dict, all_utts = ds_test.get_test_items_new()
                    test_inputs=torch.stack(test_feats).to(device)
                    test_labels = label_dict[classifier_heads].to(device)

                    outputs = model(test_inputs)
                    loss = criterion(outputs, test_labels)
                    val_loss += loss.item()

                    _, preds = torch.max(outputs, 1)
                    correct_test = (preds == test_labels).sum().item()

                    val_accuracy = 100 * correct_test / len(all_utts)
                
                # scheduler
                scheduler.step(val_loss)

                # wenn val accuracy besser ist, dann speichern als bestCheckpoint
                checkpoint_filename = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pt')
                is_best = False
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    is_best = True

                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)

                save_checkpoint(epoch, model, optimizer, val_loss, checkpoint_filename, is_best)
                log.write(f"===================================================================\n")
                log.write(f"Epoch: {epoch + 1}, Test Loss: {val_loss}, Test Acc: {val_accuracy}\n")
                log.write(f"===================================================================\n")
                print(f"Epoch: {epoch + 1}, Test Loss: {val_loss}, Test Acc: {val_accuracy}")
                print(f"Saved checkpoint to {checkpoint_filename}. Best checkpoint: {is_best}", flush=True)
                
                early_stopper_acc.step(val_accuracy)
                early_stopper_loss.step(-val_loss)
                # early stopp if signs overfitting 
                if early_stopper_acc.should_stop or early_stopper_loss.should_stop:
                    print(f"Early stopping triggered at epoch {epoch + 1}", flush=True)
                    break

    def plot_training(logdir):
        plt.figure()
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.legend()
        plt.title('Loss Curve')
        plt.savefig(os.path.join(logdir, 'loss_curve.png'))

        plt.figure()
        plt.plot(train_accuracies, label='Train Acc')
        plt.plot(val_accuracies, label='Val Acc')
        plt.legend()
        plt.title('Accuracy Curve')
        plt.savefig(os.path.join(logdir, 'accuracy_curve.png'))
    
    plot_training(checkpoint_dir)

if __name__ == "__main__":
    # exampel run:
    # python3 ./classifier_train.py ./config/config_nationality.cfg --gpu 0
    parser = argparse.ArgumentParser(description='Train classifier.')
    # see config_example/example_nationality_config.cfg for config file format
    parser.add_argument('config', type=str, help='Path to the config.cfg file.')
    parser.add_argument('--gpu', type=int, default=0, help='Which GPU to use if CUDA is enabled.')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
    args = parser.parse_args()
    
    main(args)