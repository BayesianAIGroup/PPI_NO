import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from random_fields import GaussianRF
import copy
from sklearn import gaussian_process as gp
import random

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
torch.set_default_dtype(torch.float)

num_iteration = 5

class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001, time_last=True):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T in 1D
        # x could be in shape of ntrain*w*l or ntrain*T*w*l or ntrain*w*l*T in 2D
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps
        self.time_last = time_last  # if the time dimension is the last dim

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        # sample_idx is the spatial sampling mask
        if sample_idx is None:
            std = self.std + self.eps  # n
            mean = self.mean
        else:
            if self.mean.ndim == sample_idx.ndim or self.time_last:
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if self.mean.ndim > sample_idx.ndim and not self.time_last:
                std = self.std[..., sample_idx] + self.eps  # T*batch*n
                mean = self.mean[..., sample_idx]
        # x is in shape of batch*(spatial discretization size) or T*batch*(spatial discretization size)
        x = (x * std) + mean
        return x

    def to(self, device):
        if torch.is_tensor(self.mean):
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)
        else:
            self.mean = torch.from_numpy(self.mean).to(device)
            self.std = torch.from_numpy(self.std).to(device)
        return self

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


class RangeNormalizer(object):
    def __init__(self, x, low=-1.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        mymin = torch.min(x, 0)[0].view(-1)
        mymax = torch.max(x, 0)[0].view(-1)
        diff = mymax - mymin
        diff[diff <= 0] = 1.0

        self.a = (high - low) / diff
        # self.a = (high - low)/(mymax - mymin)
        self.b = -self.a * mymax + high

    def encode(self, x):
        s = x.size()
        x = x.reshape(s[0], -1)
        x = self.a * x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.b) / self.a
        x = x.view(s)
        return x

    def cuda(self):
        self.a = self.a.cuda()
        self.b = self.b.cuda()

    def cpu(self):
        self.a = self.a.cpu()
        self.b = self.b.cpu()


# loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p,
                                                          1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

class Module(torch.nn.Module):
    '''Standard module format.
    '''

    def __init__(self):
        super(Module, self).__init__()
        self.activation = None
        self.initializer = None

        self.__device = None
        self.__dtype = None

    @property
    def device(self):
        return self.__device

    @property
    def dtype(self):
        return self.__dtype

    @device.setter
    def device(self, d):
        if d == 'cpu':
            self.cpu()
        elif d == 'gpu':
            self.cuda()
        else:
            raise ValueError
        self.__device = d

    @dtype.setter
    def dtype(self, d):
        if d == 'float':
            self.to(torch.float)
        elif d == 'double':
            self.to(torch.double)
        else:
            raise ValueError
        self.__dtype = d

    @property
    def Device(self):
        if self.__device == 'cpu':
            return torch.device('cpu')
        elif self.__device == 'gpu':
            return torch.device('cuda')

    @property
    def Dtype(self):
        if self.__dtype == 'float':
            return torch.float32
        elif self.__dtype == 'double':
            return torch.float64

    @property
    def act(self):
        if self.activation == 'sigmoid':
            return torch.sigmoid
        elif self.activation == 'relu':
            return torch.relu
        elif self.activation == 'tanh':
            return torch.tanh
        elif self.activation == 'elu':
            return torch.elu
        else:
            raise NotImplementedError

    @property
    def Act(self):
        if self.activation == 'sigmoid':
            return torch.nn.Sigmoid()
        elif self.activation == 'relu':
            return torch.nn.ReLU()
        elif self.activation == 'tanh':
            return torch.nn.Tanh()
        elif self.activation == 'elu':
            return torch.nn.ELU()
        else:
            raise NotImplementedError

    @property
    def weight_init_(self):
        if self.initializer == 'He normal':
            return torch.nn.init.kaiming_normal_
        elif self.initializer == 'He uniform':
            return torch.nn.init.kaiming_uniform_
        elif self.initializer == 'Glorot normal':
            return torch.nn.init.xavier_normal_
        elif self.initializer == 'Glorot uniform':
            return torch.nn.init.xavier_uniform_
        elif self.initializer == 'orthogonal':
            return torch.nn.init.orthogonal_
        elif self.initializer == 'default':
            if self.activation == 'relu':
                return torch.nn.init.kaiming_normal_
            elif self.activation == 'tanh':
                return torch.nn.init.orthogonal_
            else:
                return lambda x: None
        else:
            raise NotImplementedError
class StructureNN(Module):
    '''Structure-oriented neural network used as a general map based on designing architecture.
    '''

    def __init__(self):
        super(StructureNN, self).__init__()

    def predict(self, x, returnnp=False):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=self.Dtype, device=self.Device)
        return self(x).cpu().detach().numpy() if returnnp else self(x)
class DeepONet(StructureNN):
    '''Deep operator network.
    Input: [batch size, branch_dim + trunk_dim]
    Output: [batch size, 1]
    '''

    def __init__(self, branch_dim, trunk_dim, branch_depth=2, trunk_depth=3, width=50,
                 activation='relu'):
        super(DeepONet, self).__init__()
        self.branch_dim = branch_dim
        self.trunk_dim = trunk_dim
        self.branch_depth = branch_depth
        self.trunk_depth = trunk_depth
        self.width = width
        self.activation = activation

        self.modus = self.__init_modules()
        self.params = self.__init_params()

    def forward(self, x):
        # x_branch, x_trunk = x[..., :self.branch_dim], x[..., -self.trunk_dim:]
        x_branch, x_trunk = x[..., :1], x[..., -2:]
        x_branch = x_branch.reshape(x_branch.shape[0], -1)
        # x_branch = self.modus['Branch'](x_branch)
        for i in range(1, self.branch_depth):
            x_branch = self.modus['BrActM{}'.format(i)](self.modus['BrLinM{}'.format(i)](x_branch))
        x_branch = self.modus['BrLinM{}'.format(self.branch_depth)](x_branch)
        x_branch = self.modus['BrLinM{}'.format(self.branch_depth)](x_branch)

        x_branch = x_branch.unsqueeze(1).repeat(1, 64, 1)
        x_branch = x_branch.unsqueeze(1).repeat(1, 64, 1, 1)

        for i in range(1, self.trunk_depth):
            x_trunk = self.modus['TrActM{}'.format(i)](self.modus['TrLinM{}'.format(i)](x_trunk))

        x = torch.sum(x_branch * x_trunk, dim=-1, keepdim=True) + self.params['bias']
        x = x.squeeze()
        # return torch.sum(x_branch * x_trunk, dim=-1, keepdim=True) + self.params['bias']
        return x

    def trunk_forward(self, x):
        x_trunk = x[..., -self.trunk_dim:]

        for i in range(1, self.trunk_depth):
            x_trunk = self.modus['TrActM{}'.format(i)](self.modus['TrLinM{}'.format(i)](x_trunk))
        return x_trunk

    def branch_forward(self, x):
        x_branch = x[..., :self.branch_dim]

        for i in range(1, self.branch_depth):
            x_branch = self.modus['BrActM{}'.format(i)](self.modus['BrLinM{}'.format(i)](x_branch))
        x_branch = self.modus['BrLinM{}'.format(self.branch_depth)](x_branch)

        return x_branch

    def __init_modules(self):
        modules = nn.ModuleDict()
        # modules['Branch'] = FNN(self.branch_dim, self.width, self.branch_depth, self.width,
        #                         self.activation, self.initializer)
        if self.branch_depth > 1:
            modules['BrLinM1'] = nn.Linear(self.branch_dim, self.width, dtype=torch.float)
            modules['BrActM1'] = self.Act
            for i in range(2, self.branch_depth):
                modules['BrLinM{}'.format(i)] = nn.Linear(self.width, self.width, dtype=torch.float)
                modules['BrActM{}'.format(i)] = self.Act
            modules['BrLinM{}'.format(self.branch_depth)] = nn.Linear(self.width, self.width, dtype=torch.float)
        else:
            modules['BrLinM{}'.format(self.branch_depth)] = nn.Linear(self.branch_dim, self.width, dtype=torch.float)

        modules['TrLinM1'] = nn.Linear(self.trunk_dim, self.width, dtype=torch.float)
        modules['TrActM1'] = self.Act
        for i in range(2, self.trunk_depth):
            modules['TrLinM{}'.format(i)] = nn.Linear(self.width, self.width, dtype=torch.float)
            modules['TrActM{}'.format(i)] = self.Act
        return modules

    def __init_params(self):
        params = nn.ParameterDict()
        params['bias'] = nn.Parameter(torch.zeros([1]))
        return params

    def get_loss_2(self, renew_f, new_u):
        err = ((renew_f - new_u).square().sum() / new_u.square().sum()).sqrt()
        return err


class MLP(nn.Module):
    def __init__(self, layers):
        super(MLP, self).__init__()
        self.mlp = nn.ModuleList()
        for j in range(len(layers) - 1):
            self.mlp.append(nn.Conv2d(layers[j], layers[j + 1], 1))

    def forward(self, x):
        for i in range(len(self.mlp) - 1):
            x = self.mlp[i](x)
            x = F.gelu(x)
            # x = F.tanh(x)
        x = self.mlp[-1](x)
        return x


class FWD2d(nn.Module):
    def __init__(self, input_channel, channel_num, layers):
        super(FWD2d, self).__init__()
        self.channel_num = channel_num
        # self.conv = nn.Conv2d(input_channel, self.channel_num, kernel_size=(3,3), padding='same', stride=1,padding_mode='zeros')
        self.conv = nn.Conv2d(input_channel, self.channel_num, kernel_size=(3, 3), padding='same', stride=1,
                              padding_mode='zeros')

        self.layers = [self.channel_num] + layers + [1]
        # self.layers = [input_channel] + layers + [1]
        self.q = MLP(self.layers)

    # x: N x s x s
    def forward(self, x):
        x = self.conv(x)
        x = self.q(x)
        # x = x.squeeze()

        return x

def generate_f(N, R, t_range, x_range, length_scale, seed):
    np.random.seed(seed)
    random.seed(seed)
    tmin, tmax = t_range
    xmin, xmax = x_range
    R_x, R_t = R
    ls_x, ls_t = length_scale
    x = np.linspace(xmin, xmax, R_x)
    t = np.linspace(tmin, tmax, R_t)
    K_xx = gp.kernels.RBF(length_scale=ls_x)
    K_xx = K_xx(x[:, None])
    Lx =  np.linalg.cholesky(K_xx + 1e-13*np.eye(R_x))
    K_tt = gp.kernels.RBF(length_scale=ls_t)
    K_tt = K_tt(t[:, None])
    Lt = np.linalg.cholesky(K_tt + 1e-13*np.eye(R_t))
    F = np.random.randn(N, R_x, R_t)
    F = (Lx@F)@Lt.T
    return x, t, F

def get_data(Ntr, file_number, with_sampled):
    data_train = np.load(f'advection_64_data/train/advection_64_{Ntr}_train_{file_number}.npy', allow_pickle=True)
    data_test = np.load(f'advection_64_data/test/advection_64_test.npy', allow_pickle=True)
    # print(data_train[0].shape)
    f_tr = data_train[0][0:Ntr]
    f_te = data_test[0][-100:]
    u_tr = data_train[1][0:Ntr]
    u_te = data_test[1][-100:]
    sampled_num = 0
    if with_sampled:
        u_sampled_train = np.load('uaug_fno.npy', allow_pickle=True)
        f_sampled_train = np.load('faug_fno.npy', allow_pickle=True)

        u = np.concatenate([u_tr, u_sampled_train, u_te], axis=0)  # (Ntr+Nte)x 256 x 256
        f = np.concatenate([f_tr, f_sampled_train, f_te], axis=0)
        sampled_num = len(u_sampled_train)
    else:
        u = np.concatenate([u_tr, u_te], axis=0)  # (Ntr+Nte)x 256 x 256
        f = np.concatenate([f_tr, f_te], axis=0)
    ux = np.gradient(u, axis=1)
    uxx = np.gradient(ux, axis=1)
    uy = np.gradient(u, axis=2)
    uyy = np.gradient(uy, axis=2)
    grid = np.meshgrid(np.linspace(0, 1, u_tr.shape[1]), np.linspace(0, 1, u_tr.shape[2]))
    x_cor = np.expand_dims(grid[0], axis=0)  # 1 x 256 x 256
    y_cor = np.expand_dims(grid[1], axis=0)
    x_cor = np.repeat(x_cor, u.shape[0], axis=0)
    y_cor = np.repeat(y_cor, u.shape[0], axis=0)
    X = np.stack((u, ux, uxx, uy, uyy, x_cor, y_cor), axis=-1)  # (Ntr+Nte) x 256 x 256
    Xtr = X[0:Ntr + sampled_num]
    Xte = X[-100:]
    ytr = f[0:Ntr + sampled_num]
    yte = f[-100:]

    return Xtr, ytr, Xte, yte


def load_eq_simple_data(Ntr, file_number):
    data_train = np.load(f'advection_64_data/train/advection_64_{Ntr}_train_{file_number}.npy', allow_pickle=True)
    data_test = np.load(f'advection_64_data/test/advection_64_test.npy', allow_pickle=True)
    x_train = data_train[0][0:Ntr]
    x_test = data_test[0][-100:]
    y_train = data_train[1][0:Ntr]
    y_test = data_test[1][-100:]
    return x_train, y_train, x_test, y_test


def pre_train_fu(Ntr, file_number, relative_fno_model_directory):
    x_train, y_train, x_test, y_test = load_eq_simple_data(Ntr, file_number)

    train_data_number = len(x_train)
    test_data_number = len(x_test)

    batch_size = 10
    if Ntr < 10:
        batch_size = 5
    learning_rate = 0.001
    epochs = 300
    step_size = 100
    gamma = 0.5
    modes = 12
    width = 32
    s = 64
    train_loss_list = []
    test_loss_list = []

    y_train = torch.tensor(y_train, device="cuda", dtype=torch.float)
    y_test = torch.tensor(y_test, device="cuda", dtype=torch.float)
    x_train = torch.tensor(x_train, device="cuda", dtype=torch.float)
    x_test = torch.tensor(x_test, device="cuda", dtype=torch.float)

    x_normalizer = UnitGaussianNormalizer(x_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)

    y_normalizer = UnitGaussianNormalizer(y_train)
    y_train = y_normalizer.encode(y_train)

    grids = [np.linspace(0, 1, s), np.linspace(0, 1, s)]
    grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
    grid = grid.reshape(1, s, s, 2)
    grid = torch.tensor(grid, dtype=torch.float, device="cuda")
    x_train = torch.cat([x_train.reshape(train_data_number, s, s, 1), grid.repeat(train_data_number, 1, 1, 1)], dim=3)
    x_test = torch.cat([x_test.reshape(test_data_number, s, s, 1), grid.repeat(test_data_number, 1, 1, 1)], dim=3)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size,
                                              shuffle=False)

    branch_dim = 64 * 64
    # branch_dim =1
    trunk_dim = 2

    model_fu = DeepONet(branch_dim, trunk_dim).float().cuda()

    optimizer_fu = torch.optim.Adam(model_fu.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler_fu = torch.optim.lr_scheduler.StepLR(optimizer_fu, step_size=step_size, gamma=gamma)

    myloss = LpLoss(size_average=False)
    y_normalizer.cuda()
    best_err = None
    best_model = None

    total_params = sum(p.numel() for p in model_fu.parameters() if p.requires_grad)
    print(f"Number of parameters: {total_params}")

    for ep in range(epochs):
        model_fu.train()
        train_mse = 0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            optimizer_fu.zero_grad()
            x = x.to(torch.float)
            out = model_fu(x.contiguous())
            out = y_normalizer.decode(out)
            y = y_normalizer.decode(y)
            loss_fu = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
            loss_fu.backward()
            optimizer_fu.step()
            train_mse += loss_fu.item()

        scheduler_fu.step()

        model_fu.eval()
        abs_err = 0.0
        rel_err = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
                x = x.to(torch.float32)
                out = y_normalizer.decode(model_fu(x))

                rel_err += myloss(out.reshape(batch_size, -1), y.reshape(batch_size, -1)).item()

        train_mse /= train_data_number
        rel_err /= test_data_number
        train_loss_list.append(train_mse)
        test_loss_list.append(rel_err)
        if best_err is None or best_err > rel_err:
            best_err = rel_err
            best_model = copy.deepcopy(model_fu)
        print(f'fu Epoch [{ep + 1}/{epochs}], Training Loss: {train_mse:.4f}, Test Loss: {rel_err:.4f}')
    print(test_loss_list)

    f_aug = y_normalizer.decode(best_model(x_test))  # Naug*64*64
    f_aug = f_aug.reshape(test_data_number, s, s)
    error_pred = myloss(f_aug.view(test_data_number, -1), y_test.view(test_data_number, -1)) / test_data_number
    print("pred_error ", error_pred)
    f_aug_fno = f_aug.cpu().detach().numpy().reshape(test_data_number, s, s)
    np.save(f'pre_res/advection_64_predict_deeponet_res_{Ntr}_{file_number}.npy', f_aug_fno)

    #file_path = os.path.join(relative_DeOnet_model_directory, f'nl_diffusion_128_{Ntr}_DeOnet_model_{file_number}.pth')
    file_path = 'deeponet_advection_64_f_2_u_model.pth'
    torch.save({
        'model_state_dict': model_fu.state_dict(),
    }, file_path)

    return best_err


def pre_train_uf(Ntr, file_number, with_sampled):
    Xtr, Ytr, Xte, Yte = get_data(Ntr, file_number, with_sampled)
    Nte = 100
    batch_size = 10
    channel_num = 64
    NN_layers = [30, 30, 30, 30]
    epochs = 4000

    x_train = torch.tensor(Xtr, dtype=torch.float, device='cuda')
    y_train = torch.tensor(Ytr, dtype=torch.float, device='cuda')
    x_test = torch.tensor(Xte, dtype=torch.float, device='cuda')
    y_test = torch.tensor(Yte, dtype=torch.float, device='cuda')

    lb = x_train.reshape([-1, x_train.shape[-1]]).min(0).values
    ub = x_train.reshape([-1, x_train.shape[-1]]).max(0).values
    x_train = 2.0 * (x_train - lb) / (ub - lb) - 1.0
    x_test = 2.0 * (x_test - lb) / (ub - lb) - 1.0

    x_train = x_train.permute(0, 3, 1, 2)
    x_test = x_test.permute(0, 3, 1, 2)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size,
                                              shuffle=False)

    model_uf = FWD2d(7, channel_num, NN_layers).float().cuda()
    learning_rate = 1e-3
    optimizer_uf = torch.optim.Adam(model_uf.parameters(), lr=learning_rate, weight_decay=1e-4)
    myloss = LpLoss(size_average=False)

    for ep in range(epochs):
        model_uf.train()
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            optimizer_uf.zero_grad()
            out = model_uf(x)
            loss_uf = myloss(out.reshape(batch_size, -1), y.reshape(batch_size, -1))
            loss_uf.backward()
            optimizer_uf.step()
            train_l2 += loss_uf.item()
        model_uf.eval()
        test_l2 = 0.0
        loss_test = 0.0
        test_y_norm = 0.0
        test_l2_g = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
                out = model_uf(x.contiguous())
                diff = out.reshape(batch_size, -1) - y.reshape(batch_size, -1)
                test_l2 += (((diff ** 2).sum(1) / ((y.reshape(batch_size, -1) ** 2).sum(1))) ** 0.5).sum()
                loss_test += myloss(out.reshape(batch_size, -1), y.reshape(batch_size, -1))
                test_l2_g += (diff ** 2).sum()
                test_y_norm += (y ** 2).sum()
        train_l2 /= Ntr
        test_l2 /= Nte
        loss_test /= Nte
        test_l2_g = (test_l2_g / test_y_norm) ** 0.5
        print(ep, 'train l2=', train_l2, 'test l2=', test_l2, 'test loss =', loss_test, 'global loss=', test_l2_g)

    torch.save({
        'model_state_dict': model_uf.state_dict(),
        'lb': lb,
        'ub': ub,
        'xtr': x_train,
        'ytr': y_train,
        'xte': x_test,
        'yte': y_test,
    }, 'advection_64_u_2_f_model.pth')


def prepare_data(Ntr, file_num, s, normalizer_class, gen_num):
    x_train, y_train, x_test, y_test = load_eq_simple_data(Ntr, file_num)

    device = torch.device('cuda')
    x_train, y_train = torch.tensor(x_train, device=device), torch.tensor(y_train, device=device)
    x_test, y_test = torch.tensor(x_test, device=device), torch.tensor(y_test, device=device)

    x_normalizer = normalizer_class(x_train)
    y_normalizer = normalizer_class(y_train)

    x_train, x_test = map(x_normalizer.encode, (x_train, x_test))

    grid = create_grid(s)
    x_train, x_test = [torch.cat([x.reshape(len(x), s, s, 1), grid.repeat(len(x), 1, 1, 1)], dim=3) for x in
                       (x_train, x_test)]

    #_, _, f_ori = generate_f(gen_num, [128, 128], [0, 1], [-1, 1], [0.2, 0.2], 123)
    f_ori = np.load(f'advection_64_data/train/advection_64_sampled_f.npy', allow_pickle=True)
    f_ori = torch.tensor(f_ori[:gen_num], device="cuda", dtype=torch.float)
    f_ori = f_ori.cuda()
    f = x_normalizer.encode(f_ori)
    f = torch.cat([f.reshape(gen_num, s, s, 1), grid.repeat(gen_num, 1, 1, 1)], dim=3)
    f = f.to(torch.float)

    return x_train, y_train, x_test, y_test, x_normalizer, y_normalizer, f_ori, f


def create_grid(s):
    grids = np.linspace(0, 1, s)
    grid = np.vstack([xx.ravel() for xx in np.meshgrid(grids, grids)]).T
    grid = grid.reshape(1, s, s, 2)
    return torch.tensor(grid, dtype=torch.float, device='cuda')


def train_PPI_f_u_model(Ntr, loop_num, pre_fno_error_list, ppi_fno_error_list, epochs, lam, batch_size, model_directory,
                        is_fixing_uf, file_num):
    gen_num = 200

    learning_rate = 1e-3
    step_size = 100
    gamma = 0.5
    modes = 12
    width = 32
    s = 64

    # use only one file for experiment

    channel_num = 64
    NN_layers = [30, 30, 30, 30]


    batch_size = 5 if Ntr < 10 else batch_size

    if loop_num == 0:
        pretrain_error = pre_train_fu(Ntr, file_num, model_directory)
        pre_fno_error_list.append(pretrain_error)
        pre_train_uf(Ntr, file_num, False)

    # load uf
    model_uf = FWD2d(7, channel_num, NN_layers).float().cuda()
    checkpoint_uf = torch.load('advection_64_u_2_f_model.pth')
    model_uf.load_state_dict(checkpoint_uf['model_state_dict'])
    lb = checkpoint_uf['lb']
    ub = checkpoint_uf['ub']
    xtr = checkpoint_uf['xtr']
    ytr = checkpoint_uf['ytr']
    xte = checkpoint_uf['xte']
    yte = checkpoint_uf['yte']
    # load fu
    branch_dim = 64 * 64
    trunk_dim = 2

    model_fu = DeepONet(branch_dim, trunk_dim).float().cuda()
    checkpoint_fu = torch.load('deeponet_advection_64_f_2_u_model.pth')
    model_fu.load_state_dict(checkpoint_fu['model_state_dict'])

    x_train, y_train, x_test, y_test, x_normalizer, y_normalizer, sampled_f_ori, sampled_f = prepare_data(Ntr,
                                                                                                          file_num,
                                                                                                          s,
                                                                                                          UnitGaussianNormalizer,
                                                                                                          gen_num)
    y_normalizer.cuda()
    train_data_number = x_train.shape[0]
    test_data_number = x_test.shape[0]
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size,
                                              shuffle=False)
    if is_fixing_uf:
        optimizer = torch.optim.Adam(model_fu.parameters(), lr=learning_rate, weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(model_uf.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    best_err = None
    best_model = None

    myloss = LpLoss(size_average=False)

    for ep in range(epochs):
        if is_fixing_uf:
            model_fu.train()
        else:
            model_uf.train()

        train_mse = 0
        u1 = y_normalizer.decode(model_fu(sampled_f))
        ux = torch.gradient(u1, axis=1)
        uxx = torch.gradient(ux[0], axis=1)
        uy = torch.gradient(u1, axis=2)
        uyy = torch.gradient(uy[0], axis=2)

        grid = np.meshgrid(np.linspace(0, 1, u1.shape[1]), np.linspace(0, 1, u1.shape[2]))
        x_cor = np.expand_dims(grid[0], axis=0)  # 1 x 256 x 256
        y_cor = np.expand_dims(grid[1], axis=0)
        x_cor = np.repeat(x_cor, u1.shape[0], axis=0)
        y_cor = np.repeat(y_cor, u1.shape[0], axis=0)
        x_cor = torch.tensor(x_cor, dtype=torch.float, device='cuda')
        y_cor = torch.tensor(y_cor, dtype=torch.float, device='cuda')
        X_aug = torch.stack((u1, ux[0], uxx[0], uy[0], uyy[0], x_cor, y_cor), axis=-1)

        X_aug = 2.0 * (X_aug - lb) / (ub - lb) - 1.0
        X_aug = X_aug.permute(0, 3, 1, 2).to(torch.float)

        f_renew = model_uf(X_aug)

        optimizer.zero_grad()

        if is_fixing_uf:
            out = model_fu(x_train.to(torch.float))
            out = y_normalizer.decode(out)
            loss1 = myloss(out.reshape(train_data_number, -1),
                           y_train.reshape(train_data_number, -1)) / train_data_number
        else:
            f_pred = model_uf(xtr)
            loss1 = myloss(f_pred.reshape(train_data_number, -1),
                           ytr.reshape(train_data_number, -1)) / train_data_number
        loss2 = lam * myloss(f_renew.reshape(gen_num, -1), sampled_f_ori.reshape(gen_num, -1)) / gen_num
        loss = loss1 + loss2

        loss.backward()

        optimizer.step()
        train_mse += loss.item()
        # scheduler.step()

        if is_fixing_uf:
            model_fu.eval()
            rel_err = 0.0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.cuda(), y.cuda()
                    x = x.to(torch.float32)
                    out = y_normalizer.decode(model_fu(x))
                    err = myloss(out.reshape(batch_size, -1), y.reshape(batch_size, -1)).item()
                    rel_err += err
        else:
            model_uf.eval()
            rel_err = 0.0
            with torch.no_grad():
                y_pred = model_uf(xte)
                err = myloss(y_pred.reshape(100, -1), yte.reshape(100, -1)).item()
                rel_err += err

        train_mse /= train_data_number
        rel_err /= test_data_number
        if best_err is None or best_err > rel_err:
            best_err = rel_err
            # best_model = copy.deepcopy(model_fu)
        # train_loss_list.append(train_mse)
        # test_loss_list.append(rel_err)
        print(f'new fu Epoch [{ep + 1}/{epochs}], Training Loss: {train_mse:.4f}, Test Loss: {rel_err:.4f}')

        total_params = sum(p.numel() for p in model_fu.parameters() if p.requires_grad)
        print(f"Number of parameters: {total_params}")

        total_params2 = sum(p.numel() for p in model_uf.parameters() if p.requires_grad)
        print(f"Number of parameters: {total_params2}")

    file_path = 'deeponet_advection_64_f_2_u_model.pth'
    uf_file_path = 'advection_64_u_2_f_model.pth'
    if is_fixing_uf:
        torch.save({
            'model_state_dict': model_fu.state_dict(),
        }, file_path)
    else:
        torch.save({
            'model_state_dict': model_uf.state_dict(),
            'lb': lb,
            'ub': ub,
            'xtr': xtr,
            'ytr': ytr,
            'xte': xte,
            'yte': yte,
        }, uf_file_path)
    print("best error", best_err)
    if loop_num == num_iteration - 1:
        ppi_fno_error_list.append(best_err)
    # if file_num == 5 and loop_num == 2:
    #     predict_result = y_normalizer.decode(model_fu(x_test))
    #     predict_result_cpu = predict_result.cpu().detach().numpy()
    #     np.save(f'advection_64_data/pre_res/advection_64_ppi_deeponet_predict_res_{Ntr}_{file_num}.npy', predict_result_cpu)
    #     del predict_result
    #     del predict_result_cpu


def setup_directories(base_path='advection_64_data/fno_model'):
    current_directory = os.getcwd()
    model_directory = os.path.join(current_directory, base_path)
    os.makedirs(model_directory, exist_ok=True)
    return model_directory


if __name__ == '__main__':
    model_directory = setup_directories()

    Ntr_list = [20,30,50,80]  # Define your training sizes
    file_list = [5]
    epochs = 1000
    batch_size = 10
    lam = 20
    total_mean_record = []
    total_std_record = []
    pre_fno_mean_record = []
    pre_fno_std_record = []
    for Ntr in Ntr_list:
        cum_err = []
        cum_pre_fno_error = []
        for file_num in file_list:
            torch.manual_seed(0)
            np.random.seed(0)

            for i in range(num_iteration):
                if i % 2 == 0:
                    is_fixing_uf = True
                    lam = 5
                    epochs = 600
                else:
                    is_fixing_uf = False
                    lam = 0.2
                    epochs = 500
                train_PPI_f_u_model(Ntr, i, cum_pre_fno_error, cum_err, epochs=epochs, lam=lam, batch_size=batch_size,
                                    model_directory=model_directory, is_fixing_uf=is_fixing_uf, file_num=file_num)
                torch.manual_seed(0)
                np.random.seed(0)

        print(len(cum_err), len(cum_pre_fno_error))
        error_mean = np.mean(cum_err)
        error_standard_deviation = np.std(cum_err) / np.sqrt(5)
        pre_fno_error_mean = np.mean(cum_pre_fno_error)
        pre_fno_error_standard_deviation = np.std(cum_pre_fno_error) / np.sqrt(5)
        total_mean_record.append(error_mean)
        total_std_record.append(error_standard_deviation)
        pre_fno_mean_record.append(pre_fno_error_mean)
        pre_fno_std_record.append(pre_fno_error_standard_deviation)

    print(len(total_mean_record),len(total_std_record),len(pre_fno_mean_record),len(pre_fno_std_record))
    print("20,30,50,80 advection_64 deeponet_pretrain mean error:", pre_fno_mean_record)
    print("20,30,50,80 advection_64 deeponet_pretrain std error:", pre_fno_std_record)
    print("20,30,50,80 advection_64 joint_training mean error:", total_mean_record)
    print("20,30,50,80 advection_64 joint_training std error:", total_std_record)
