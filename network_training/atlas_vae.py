from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
from glob import glob
from util import *
import numpy as np
from PIL import Image


totensor = transforms.ToTensor()


def load_datasets(trn_path, tst_path, data_name, b_size):
    training_data = CustomImageDataset(trn_path, data_name)
    test_data = CustomImageDataset(tst_path, data_name)

    training_loader = DataLoader(training_data, batch_size=b_size, shuffle=True)
    testing_loader = DataLoader(test_data, batch_size=b_size, shuffle=True)
    return training_loader, testing_loader


class VAE(nn.Module):
    def __init__(self, nc, ngf, ndf, latent_variable_size, batch_size,  image_size, gpu=False):
        super(VAE, self).__init__()

        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.img_sz = image_size
        self.latent_variable_size = latent_variable_size
        self.batch_size = batch_size
        self.use_gpu = gpu

        # encoder
        # self.e1 = nn.Conv2d(nc, ndf, (4, 4), stride=2, padding=1)
        # self.bn1 = nn.BatchNorm2d(ndf)
        #
        # self.e2 = nn.Conv2d(ndf, ndf * 2, (4, 4), stride=2, padding=1)
        # self.bn2 = nn.BatchNorm2d(ndf * 2)
        #
        # self.e3 = nn.Conv2d(ndf * 2, ndf * 4, (4, 4), stride=2, padding=1)
        # self.bn3 = nn.BatchNorm2d(ndf * 4)
        #
        # self.e4 = nn.Conv2d(ndf * 4, ndf * 8, (4, 4), stride=2, padding=1)
        # self.bn4 = nn.BatchNorm2d(ndf * 8)
        self.e1 = nn.Conv2d(nc, ndf, (4, 4), stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(ndf)

        self.e2 = nn.Conv2d(ndf, ndf * 2, (4, 4), stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(ndf * 2)

        self.e3 = nn.Conv2d(ndf * 2, ndf * 4, (4, 4), stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(ndf * 4)

        self.e4 = nn.Conv2d(ndf * 4, ndf * 8, (4, 4), stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(ndf * 8)

        if self.img_sz == 32:
            self.fc1 = nn.Linear(ndf * 8 * 4, latent_variable_size)
            self.fc2 = nn.Linear(ndf * 8 * 4, latent_variable_size)
        elif self.img_sz == 64:
            self.fc1 = nn.Linear(ndf * 8 * 4 * 4, latent_variable_size)
            self.fc2 = nn.Linear(ndf * 8 * 4 * 4, latent_variable_size)

        # decoder
        if self.img_sz == 32:
            self.d1 = nn.Linear(latent_variable_size, 4 * ngf * 8)
        elif self.img_sz == 64:
            self.d1 = nn.Linear(latent_variable_size, 4 * 4 * ngf * 8)

        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.d2 = nn.Conv2d(ngf * 8, ngf * 4, (3, 3), stride=1, padding='same')
        self.bn6 = nn.BatchNorm2d(ngf * 4, 1.e-3)

        # self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        # self.pd2 = nn.ReplicationPad2d(1)
        # self.d3 = nn.Conv2d(ngf * 8, ngf * 4, 3, 1)
        # self.bn7 = nn.BatchNorm2d(ngf * 4, 1.e-3)

        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.d3 = nn.Conv2d(ngf * 4, ngf * 2, (3, 3), stride=1, padding='same')
        self.bn7 = nn.BatchNorm2d(ngf * 2, 1.e-3)

        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.d4 = nn.Conv2d(ngf * 2, ngf, (3, 3), stride=1, padding='same')
        self.bn8 = nn.BatchNorm2d(ngf, 1.e-3)

        self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.d5 = nn.Conv2d(ngf, nc, (3, 3), stride=1, padding='same')

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.leakyrelu(self.bn1(self.e1(x)))
        h2 = self.leakyrelu(self.bn2(self.e2(h1)))
        h3 = self.leakyrelu(self.bn3(self.e3(h2)))
        h4 = self.leakyrelu(self.bn4(self.e4(h3)))
        h5 = h4.flatten(start_dim=1)
        return self.fc1(h5), self.fc2(h5)

    def decode(self, z):
        h1 = self.relu(self.d1(z))
        if self.img_sz == 32:
            h1 = h1.view(-1, self.ngf * 8, 2, 2)  # Network reshaping
        elif self.img_sz == 64:
            h1 = h1.view(-1, self.ngf * 8, 4, 4)
        h2 = self.leakyrelu(self.bn6(self.d2(self.up1(h1))))
        h3 = self.leakyrelu(self.bn7(self.d3(self.up2(h2))))
        h4 = self.leakyrelu(self.bn8(self.d4(self.up3(h3))))

        return self.sigmoid(self.d5(self.up4(h4)))

    def reparametrize(self, mu, logvar):
        """Converts the mean and standard deviation into a single vector"""
        std = logvar.mul(0.5).exp_()
        if self.use_gpu:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        # eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def get_latent_var(self, x):
        """Returns the latent variable from an input image"""
        # mu, logvar = self.encode(x.view(-1, self.nc, self.img_sz, self.img_sz))
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return z

    def forward(self, x):
        """Runs the model in a forward fashion. Returns reconstruction, mean, and variance"""
        # print("Forward shape", np.shape(x))
        # mu, logvar = self.encode(x.view(-1, self.nc, self.img_sz, self.img_sz))
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        res = self.decode(z)
        return res, mu, logvar

    def apply_model(self, x):
        """Function used to extract latent variable and reconstructed image"""
        # print("Forward shape", np.shape(x))
        # mu, logvar = self.encode(x.view(-1, self.nc, self.img_sz, self.img_sz))
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        res = self.decode(z)
        return res, z


def loss_function(recon_x, x, mu, logvar):
    # print(np.shape(recon_x))
    MSE = reconstruction_function(recon_x, x)

    # https://arxiv.org/abs/1312.6114 (Appendix B)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)

    return MSE + args.beta*KLD


def train(epoch, loader):
    model.train()
    train_loss = 0
    batch_idx = 0
    for data in loader:
        # data = totensor(data)
        if torch.cuda.is_available():
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        # print("Loss:", loss)
        loss.backward()
        # print("Loss:", loss.data)
        train_loss += loss.data
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), (loader.__len__() * model.batch_size),
                       100. * batch_idx /loader.__len__(),
                       loss.data / len(data)))

            # Calculate gradients
            parameters = model.parameters()
            gradients = []
            for param in filter(lambda param: param.grad is not None, parameters):
                gradients.append(torch.mean(torch.abs(param.grad.data)))
            writer.add_scalar('Loss/train', loss.data, batch_idx*epoch)
            writer.add_scalar("gradient", torch.mean(torch.stack(gradients)), batch_idx*epoch)
            writer.close()
        batch_idx += 1

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / (loader.__len__() * model.batch_size)))
    return train_loss / (loader.__len__() * model.batch_size)


def test(epoch, loader):
    model.eval()
    test_loss = 0
    for data in loader:
        with torch.no_grad():
            if torch.cuda.is_available():
                data = data.cuda()
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).data
            data_image = torchvision.utils.make_grid(data.data, nrow=8, padding=2)
            recon_image = torchvision.utils.make_grid(recon_batch.data, nrow=8, padding=2)
            try:
                torchvision.utils.save_image(data_image, '../output/{}Epoch_{}_data.jpg'.format(logging_comment, epoch))
                torchvision.utils.save_image(recon_image, '../output/{}Epoch_{}_recon.jpg'.format(logging_comment, epoch))
            except FileNotFoundError:
                torchvision.utils.save_image(data_image, './output/{}Epoch_{}_data.jpg'.format(logging_comment, epoch))
                torchvision.utils.save_image(recon_image, './output/{}Epoch_{}_recon.jpg'.format(logging_comment, epoch))

    test_loss /= (loader.__len__() * model.batch_size)
    writer.add_scalar('Loss/test', test_loss, epoch)
    writer.close()
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss


def perform_latent_space_arithmatics(items):  # input is list of tuples of 3 [(a1,b1,c1), (a2,b2,c2)]
    load_last_model()
    model.eval()
    data = [im for item in items for im in item]
    data = [totensor(i) for i in data]
    data = torch.stack(data, dim=0)
    data = Variable(data, volatile=True)
    if torch.cuda.is_available():
        data = data.cuda()
    z = model.get_latent_var(data.view(-1, model.nc, model.ndf, model.ngf))
    it = iter(z.split(1))
    z = zip(it, it, it)
    zs = []
    numsample = 11
    for i, j, k in z:
        for factor in np.linspace(0, 1, numsample):
            zs.append((i - j) * factor + k)
    z = torch.cat(zs, 0)
    recon = model.decode(z)

    it1 = iter(data.split(1))
    it2 = [iter(recon.split(1))] * numsample
    result = zip(it1, it1, it1, *it2)
    result = [im for item in result for im in item]

    result = torch.cat(result, 0)
    torchvision.utils.save_image(result.data, '../output/vec_math.jpg', nrow=3 + numsample, padding=2)


def latent_space_transition(model_name, items):  # input is list of tuples of  (a,b)
    state_dict = torch.load(model_name)
    model.load_state_dict(state_dict)
    model.eval()
    data = [im for item in items for im in item[:-1]]
    # data = [totensor(i) for i in data]
    data = torch.stack(data, dim=0)
    # data = Variable(data)
    if args.cuda:
        data = data.cuda()
    z = model.get_latent_var(data.view(-1, model.nc, model.ndf, model.ngf))
    it = iter(z.split(1))
    z = zip(it, it)
    zs = []
    numsample = 6
    for i, j in z:
        for factor in np.linspace(0, 1, numsample):
            zs.append(i + (j - i) * factor)
    z = torch.cat(zs, 0)
    recon = model.decode(z)

    it1 = iter(data)
    it2 = [iter(recon.split(1))] * numsample
    result = zip(it1, it1, *it2)
    result = [im for item in result for im in item]
    result = torch.cat(result, 0)

    total_images = int(np.ceil(65500/np.shape(result)[0]))
    batch = np.shape(result)[0]
    b = 0
    img_set = 30
    torch_grid = torchvision.utils.make_grid(result.data, nrow=2+numsample, padding=1)
    transform = torchvision.transforms.ToPILImage()
    PIL_grid = transform(torch_grid)
    PIL_grid.save('../output/trans.tiff', compression=None)
    # for i in range(batch-img_set):
    #     if i > batch:
    #         torchvision.utils.save_image(result.data[i:, :, :, :],
    #                                      '../output/trans{}.jpg'.format(b), nrow=2 + numsample, padding=2)
    #     else:
    #         torchvision.utils.save_image(result.data[i:i+img_set, :, :, :],
    #                                      '../output/trans{}.jpg'.format(b), nrow=2 + numsample, padding=2)
    #     b += 1


def load_last_model():
    models = glob('../network_training/models/*.pth')
    # print(models)
    model_ids = [(int(f.split('_')[2]), f) for f in models]
    # print(model_ids)
    start_epoch, last_cp = max(model_ids, key=lambda item: item[0])
    print('Last checkpoint: ', last_cp)
    model.load_state_dict(torch.load(last_cp))
    return start_epoch, last_cp


def engage_training(resume, n_path, s_path, data_name, batch):
    if resume:
        start_epoch, _ = load_last_model()
    else:
        start_epoch = 0
    # Setup Loaders
    trainer, tester = load_datasets(n_path, s_path, batch, data_name)
    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        train_loss = train(epoch, trainer)
        test_loss = test(epoch, tester)
        scheduler.step()
        try:
            torch.save(model.state_dict(),
                       '../network_training/models/{}_Epoch_{}_Train_loss_{:.4f}_Test_loss_{:.4f}.pth'
                       .format(logging_comment, epoch, train_loss, test_loss))
        except FileNotFoundError:
            torch.save(model.state_dict(),
                       './network_training/models/{}_Epoch_{}_Train_loss_{:.4f}_Test_loss_{:.4f}.pth'
                       .format(logging_comment, epoch, train_loss, test_loss))


def last_model_to_cpu():
    _, last_cp = load_last_model()
    model.cpu()
    torch.save(model.state_dict(), '../models/cpu_' + last_cp.split('/')[-1])


# Training design #################################################################################################
if __name__ == '__main__':

    bite_size = 128         # Batch Size
    nerve_factor = 32       # Nerve growth factor
    LVS_size = 150          # Size of latent variable
    learning_rate = 1e-4    # Starting learning rate
    gamma = 0.6             # Exponential decay constant for learning rate
    logging_rate = 10       # Rate at which information is logged
    epochs = 40             # Number of Epochs
    beta = .1               # Beta KLD reconstruction weighting

    parser = argparse.ArgumentParser(description='PyTorch VAE')
    parser.add_argument('--batch', type=int, default=bite_size,
                        help='input batch size for training (default: {})'.format(bite_size))
    parser.add_argument('--lr', type=float, default=learning_rate,
                        help='starting learning rate (default: {})'.format(learning_rate))
    parser.add_argument('--epochs', type=int, default=epochs,
                        help='number of epochs to train (default: {})'.format(epochs))
    parser.add_argument('--nerve', type=int, default=nerve_factor,
                        help='nerve growth and decay factor (default: {}'.format(nerve_factor))
    parser.add_argument('--lvs', type=int, default=LVS_size,
                        help='size of latent variable (default: {})'.format(LVS_size))
    parser.add_argument('--log-interval', type=int, default=logging_rate,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--gamma', type=float, default=gamma,
                        help='exponential decay of learning rate (default: {})'.format(gamma))
    parser.add_argument('--beta', type=float, default=beta,
                        help='weighting constant for KLD reconstruction loss (default: {})'.format(beta))
    parser.add_argument('--com', type=str, default='',
                        help='optional extra comment for logging')
    args = parser.parse_args()

    # args.cuda = torch.cuda.is_available()
    args.cuda = True
    args.log_interval = logging_rate

    model = VAE(nc=3, ngf=args.nerve, ndf=args.nerve,
                latent_variable_size=args.lvs, batch_size=args.batch, image_size=64, gpu=args.cuda)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, eps=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

    logging_comment = str(args.com+"_ExpLR{}_{}_LVS{}_nf{}_b{}".format(args.gamma, args.lr, args.lvs, args.nerve, args.beta))
    if args.cuda:
        model.cuda()
        print("Training with GPU")
    print(logging_comment)

    # Set random seed
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    reconstruction_function = nn.MSELoss()
    reconstruction_function.reduction = 'mean'

########################################################################################################################

    # Image sampling and saving
    img_path = "../images/Boulder_flatirons_topo_FS_raw@4_NC.jpeg"
    save_path = "../../training_sample_set_FS_60k_4x_NC_32/train/"
    samples = 60000
    sample_size = 32
    name = "Eldorado_FS_"

    # Training Paths
    train_path = '../../training_sample_set_FS_60k_4x_NC/train/'
    test_path = '../../training_sample_set_FS_60k_4x_NC/test/'

    # sample_image(img_path, save_path, name, samples, sample_size)
    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter(comment=logging_comment)
    engage_training(False, train_path, test_path, args.batch, name)
    # train()
    # last_model_to_cpu()
    # load_last_model()
    # rand_faces(10)

# Test loader
#     m, s = get_image_metrics(img_path)
#     print(m, s)
#     training, testing = load_datasets(train_path, test_path, name, 64)
#     train_features = next(iter(training))
#     print(f"Feature batch shape: {train_features.size()}")
#     data_image = torchvision.utils.make_grid(train_features.data, nrow=8, padding=2)
#     torchvision.utils.save_image(data_image, '../output/testing_image_@4_NC.jpg')
    # img = train_features[0].squeeze()
    # img.show()
    # img = img.reshape(64, 64, 3)
    # plt.imshow(img)
    # plt.show()

# Test model
#     saved_reference = "saved_models/_ExpLR0.8_0.0001_LVS150_nf32_b0.1.pth"
#     it = iter(testing)
#     # print(it)
#     l = zip(it, it)
#     latent_space_transition(saved_reference, l)
#     # perform_latent_space_arithmatics(l)
