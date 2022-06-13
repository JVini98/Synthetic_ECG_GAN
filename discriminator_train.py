# ===========================================
# Reference: https://github.com/mazzzystar/WaveGAN-pytorch
# ===========================================

import argparse
import os
from tqdm import tqdm
import numpy as np

# Pytorch
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from torchvision import models, transforms
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torchsummary import summary
from torch import autograd
from torch.utils.data import random_split

# Model specific

from data.ecg_data_loader import ECGDataSimple as ecg_data
from models.discriminador import Pulse2pulseDiscriminator as Pulse2PulseDiscriminator
from utils.utils import calc_gradient_penalty, get_plots_RHTM_10s, get_plots_all_RHTM_10s

torch.manual_seed(0)
np.random.seed(0)
parser = argparse.ArgumentParser()

# Hardware
parser.add_argument("--device_id", type=int, default=0, help="Device ID to run the code")
parser.add_argument("--exp_name", type=str, required=True,
                    help="A name to the experiment which is used to save checkpoints and tensorboard output")
# parser.add_argument("--py_file",default=os.path.abspath(__file__)) # store current python file


# ==============================
# Directory and file handling
# ==============================
parser.add_argument("--data_dirs_N", default=["/home/jvini/PycharmProjects/pulse2pulse_pycharm/sample_ecg_data",
                                            ], help="Data roots", nargs="*")

parser.add_argument("--data_dirs_AF", default=["/home/jvini/PycharmProjects/pulse2pulse_pycharm/sample_ecg_data",
                                            ], help="Data roots", nargs="*")

parser.add_argument("--out_dir",
                    default="/home/jvini/PycharmProjects/pulse2pulse_pycharm/Result_testing_out/output",
                    help="Main output dierectory")

parser.add_argument("--tensorboard_dir",
                    default="/home/jvini/PycharmProjects/pulse2pulse_pycharm/Result_testing_out/tensorboard",
                    help="Folder to save output of tensorboard")
# ======================
# Hyper parameters
# ======================
parser.add_argument("--bs", type=int, default=32, help="Mini batch size")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate for training")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.9, help="adam: decay of first order momentum of gradient")

parser.add_argument("--num_epochs", type=int, default=400, help="number of epochs of training")
parser.add_argument("--start_epoch", type=int, default=0, help="Start epoch in retraining")
parser.add_argument("--ngpus", type=int, default=1, help="Number of GPUs used in models")
parser.add_argument("--checkpoint_interval", type=int, default=25, help="Interval to save checkpoint models")

# Checkpoint path to retrain or test models
parser.add_argument("--checkpoint_path", default="", help="Check point path to retrain or test models")

parser.add_argument('-ms', '--model_size', type=int, default=25,
                    help='Model size parameter used in WaveGAN')
parser.add_argument('--lmbda', default=10.0, help="Gradient penalty regularization factor")

# Action handling
parser.add_argument("action", type=str, help="Select an action to run",
                    choices=["train", "retrain", "inference", "check"])

opt = parser.parse_args()
print(opt)

# ==========================================
# Device handling
# ==========================================
torch.cuda.set_device(opt.device_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("device=", device)

# ===========================================
# Folder handling
# ===========================================

# make output folder if not exist
os.makedirs(opt.out_dir, exist_ok=True)

# make subfolder in the output folder
checkpoint_dir = os.path.join(opt.out_dir, opt.exp_name + "/checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

# make tensorboard subdirectory for the experiment
tensorboard_exp_dir = os.path.join(opt.tensorboard_dir, opt.exp_name)
os.makedirs(tensorboard_exp_dir, exist_ok=True)

# ==========================================
# Tensorboard
# ==========================================
# Initialize summary writer
writer = SummaryWriter(tensorboard_exp_dir)


# ==========================================
# Prepare Data
# ==========================================
def prepare_data():
    dataset_normal = ecg_data(opt.data_dirs_N, norm_num=600, cropping=None, transform=None)
    dataset_AF = ecg_data(opt.data_dirs_AF, norm_num=600, cropping=None, transform=None)

    dataset = []

    for i in range(len(dataset_normal)):
        dataset.append([dataset_normal[i], 0])

    for i in range(len(dataset_AF)):
        dataset.append([dataset_AF[i], 1])

    print("Dataset size=", len(dataset))

    train_dataset, test_dataset = random_split(dataset, [5025, 558])

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=opt.bs,
                                             shuffle=True,
                                             num_workers=8
                                             )

    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=len(test_dataset),
                                             shuffle=True,
                                             num_workers=8
                                             )

    return train_dataloader, test_dataloader


# ===============================================
# Prepare models
# ===============================================
def prepare_model():
    netD = Pulse2PulseDiscriminator(model_size=opt.model_size,num_channels=1 ,ngpus=opt.ngpus)

    netD = netD.to(device)

    return netD


# ====================================
# Run training process
# ====================================
def run_train():

    netD = prepare_model()

    optimizerD = optim.RMSprop(netD.parameters(), lr=opt.lr, momentum=0)

    train_dataloader, test_dataloader = prepare_data()

    lossfun = nn.BCEWithLogitsLoss()

    train(netD, optimizerD, lossfun ,train_dataloader, test_dataloader)

def extractDigits(lst):
                return list(map(lambda el:[el], lst))

def train(netD, optimizerD, lossfun, train_dataloader, test_dataloader):
    
    trainLoss = torch.zeros(opt.num_epochs)
    testLoss  = torch.zeros(opt.num_epochs)
    trainAcc  = torch.zeros(opt.num_epochs)
    testAcc   = torch.zeros(opt.num_epochs)

    
    for epoch in tqdm(range(opt.start_epoch + 1, opt.start_epoch + opt.num_epochs + 1)):

        len_dataloader_train = len(train_dataloader)
        len_dataloader_test = len(test_dataloader)
        print("Length of Dataloaders:", len_dataloader_train, len_dataloader_test)

        netD.train()

        batchAcc  = []
        batchLoss = []

        for i, sample in tqdm(enumerate(train_dataloader, 0)):

            ecgs = sample[0]["ecg_signals"].to(device)
            labels = sample[1]

            prediction = netD(ecgs)

            labels = torch.Tensor(extractDigits(labels.numpy())).to('cuda')

            loss = lossfun(prediction,labels)

            optimizerD.zero_grad()
            loss.backward()
            optimizerD.step()

            batchLoss.append(loss.item())

            norm_predictions = torch.sigmoid(prediction)

            acc = torch.mean(((norm_predictions>.5).float().detach() == labels).float())
            batchAcc.append( 100*acc.item() ) 


        trainLoss[epoch] = np.mean(batchLoss)
        trainAcc[epoch] =  np.mean(batchAcc) 

        netD.eval()
        for i, sample in tqdm(enumerate(test_dataloader, 0)):

            ecgs = sample[0]["ecg_signals"].to(device)
            labels = sample[1]
            labels = torch.Tensor(extractDigits(labels.numpy())).to('cuda')

            with torch.no_grad():
                test_prediction = netD(ecgs)
                loss = lossfun(test_prediction,labels)

        test_pred_norm = torch.sigmoid(test_prediction)

        testAcc[epoch] = 100*torch.mean(((test_pred_norm>.5).float().detach() == labels).float()).item()

        writer.add_scalar("Train_Loss", trainLoss[epoch], epoch)
        writer.add_scalar("Train_Acc", trainAcc[epoch], epoch)
        writer.add_scalar("Test_Acc", testAcc[epoch], epoch)

        print("Epochs:{}\t\ttrain_loss:{}\t\t train_acc:{}\t\ttest_acc:{}".format(
            epoch, trainLoss[epoch], trainAcc[epoch], testAcc[epoch]))

        # Save model
        if epoch % opt.checkpoint_interval == 0:
            save_model( netD, optimizerD, epoch)
            #fig = get_plots_RHTM_10s(real_ecgs_to_plot[0].detach().cpu(), fake_to_plot[0].detach().cpu())
            #fig_2 = get_plots_all_RHTM_10s(real_ecgs_to_plot.detach().cpu(), fake_to_plot.detach().cpu())

            #writer.add_figure("sample", fig, epoch)
            #writer.add_figure("sample_batch", fig_2, epoch)
        # fig.savefig("{}.png".format(epoch))


# =====================================
# Save models
# =====================================
def save_model(netG, netD, optimizerG, optimizerD, epoch, py_file_name="test"):
    check_point_name = py_file_name + "_epoch:{}.pt".format(epoch)  # get code file name and make a name
    check_point_path = os.path.join(checkpoint_dir, check_point_name)
    # save torch model
    torch.save({
        "epoch": epoch,
        "netG_state_dict": netG.state_dict(),
        "netD_state_dict": netD.state_dict(),
        "optimizerG_state_dict": optimizerG.state_dict(),
        "optimizerD_state_dict": optimizerD.state_dict(),
        # "train_loss": train_loss,
        # "val_loss": validation_loss
    }, check_point_path)


# ====================================
# Re-train process
# ====================================
def run_retrain():
    print("run retrain started........................")
    netG, netD = prepare_model()

    # netG.cpu()
    # netD.cpu()

    # loading checkpoing
    chkpnt = torch.load(opt.checkpoint_path, map_location="cpu")

    netG.load_state_dict(chkpnt["netG_state_dict"])
    netD.load_state_dict(chkpnt["netD_state_dict"])

    netG = netG.to(device)
    netD = netD.to(device)

    print("model loaded from checkpoint=", opt.checkpoint_path)

    # setup start epoch to checkpoint epoch
    opt.__setattr__("start_epoch", chkpnt["epoch"])

    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    dataloaders = prepare_data()
    train(netG, netD, optimizerG, optimizerD, dataloaders)


# =====================================
# Check model
# ====================================
def check_model_graph():
    netG, netD = prepare_model()
    print(netG)
    netG = netG.to(device)
    netD = netD.to(device)

    summary(netG, (8, 5000))
    summary(netD, (8, 5000))


if __name__ == "__main__":

    data_loaders = prepare_data()
    print(vars(opt))
    print("Test OK")

    # Train or retrain or inference
    if opt.action == "train":
        print("Training process is strted..!")
        run_train()
        pass
    elif opt.action == "retrain":
        print("Retrainning process is strted..!")
        run_retrain()
        pass
    elif opt.action == "inference":
        print("Inference process is strted..!")
        pass
    elif opt.action == "check":
        check_model_graph()
        print("Check pass")

    # Finish tensorboard writer
    writer.close()
