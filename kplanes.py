import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


class NerfModel(nn.Module):
    def __init__(self, embedding_dim_direction=4, hidden_dim=64, N=512, M=512, F=96, scale_x=1000, scale_y=380, scale_z=150):
        """
        The parameter scale represents the maximum absolute value among all coordinates and is used for scaling the data
        """
        super(NerfModel, self).__init__()

        self.xy_plane = nn.Parameter(torch.rand((N, N, F)))
        self.yz_plane = nn.Parameter(torch.rand((N, M, F)))
        self.xz_plane = nn.Parameter(torch.rand((N, M, F)))

        self.block1 = nn.Sequential(nn.Linear(F, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 16), nn.ReLU(), )
        self.block2 = nn.Sequential(nn.Linear(15 + 3 * 4 * 2 + 3, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, 3), nn.Sigmoid())

        self.embedding_dim_direction = embedding_dim_direction
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.scale_z = scale_z
        self.N = N
        self.M = M

    @staticmethod
    def positional_encoding(x, L):
        out = [x]
        for j in range(L):
            out.append(torch.sin(2 ** j * x))
            out.append(torch.cos(2 ** j * x))
        return torch.cat(out, dim=1)

    def forward(self, x, d):
        sigma = torch.zeros_like(x[:, 0])
        c = torch.zeros_like(x)

        mask = (x[:, 0].abs() < self.scale_x) & (x[:, 1].abs() < self.scale_y) & (x[:, 2].abs() < self.scale_z)
        #XYplane
        xy_idx_x = ((x[:, [0]] / (2 * self.scale_x) + .5) * self.N).long().clip(0, self.N - 1)
        xy_idx_y = ((x[:, [1]] / (2 * self.scale_y) + .5) * self.M).long().clip(0, self.M - 1)
        xy_idx = torch.cat([xy_idx_x, xy_idx_y], dim=1) # [batch_size, 2]
        #YZplane
        yz_idx_y = ((x[:, [1]] / (2 * self.scale_y) + .5) * self.N).long().clip(0, self.N - 1)
        yz_idx_z = ((x[:, [2]] / (2 * self.scale_z) + .5) * self.M).long().clip(0, self.M - 1)
        yz_idx = torch.cat([yz_idx_y, yz_idx_z], dim=1) # [batch_size, 2]
        #XZplane
        xz_idx_x = ((x[:, [1]] / (2 * self.scale_x) + .5) * self.N).long().clip(0, self.N - 1)
        xz_idx_z = ((x[:, [2]] / (2 * self.scale_z) + .5) * self.M).long().clip(0, self.M - 1)
        xz_idx = torch.cat([xz_idx_x, xz_idx_z], dim=1) # [batch_size, 2]


        F_xy = self.xy_plane[xy_idx[mask, 0], xy_idx[mask, 1]]  # [batch_size, F]
        F_yz = self.yz_plane[yz_idx[mask, 0], yz_idx[mask, 1]]  # [batch_size, F]
        F_xz = self.xz_plane[xz_idx[mask, 0], xz_idx[mask, 1]]  # [batch_size, F]
        F = F_xy * F_yz * F_xz  # [batch_size, F]

        h = self.block1(F)
        h, sigma[mask] = h[:, :-1], h[:, -1]
        c[mask] = self.block2(torch.cat([self.positional_encoding(d[mask], self.embedding_dim_direction), h], dim=1))
        return c, sigma


@torch.no_grad()
def test(hn, hf, dataset, chunk_size=8, img_index=0, nb_bins=192, H=400, W=400):
    model_trained = NerfModel(hidden_dim=256).to(device)
    model_trained.load_state_dict(torch.load('final_model.pth'))
    model_trained.eval()

    ray_origins = dataset[img_index * H * W: (img_index + 1) * H * W, :3]
    ray_directions = dataset[img_index * H * W: (img_index + 1) * H * W, 3:6]

    data = []
    for i in range(int(np.ceil(H / chunk_size))):
        ray_origins_ = ray_origins[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
        ray_directions_ = ray_directions[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
        regenerated_px_values = render_rays(model_trained, ray_origins_, ray_directions_, hn=hn, hf=hf, nb_bins=nb_bins)
        data.append(regenerated_px_values)
    img = torch.cat(data).data.cpu().numpy().reshape((H, W, 3))

    plt.figure()
    plt.imshow(img)
    plt.savefig(f'novel_views/img_{img_index}.png', bbox_inches='tight')
    plt.close()


def compute_accumulated_transmittance(alphas):
    accumulated_transmittance = torch.cumprod(alphas, 1)
    return torch.cat((torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
                      accumulated_transmittance[:, :-1]), dim=-1)


def render_rays(nerf_model, ray_origins, ray_directions, hn=0, hf=0.5, nb_bins=192):
    device = ray_origins.device
    t = torch.linspace(hn, hf, nb_bins, device=device).expand(ray_origins.shape[0], nb_bins)
    # Perturb sampling along each ray.
    mid = (t[:, :-1] + t[:, 1:]) / 2.
    lower = torch.cat((t[:, :1], mid), -1)
    upper = torch.cat((mid, t[:, -1:]), -1)
    u = torch.rand(t.shape, device=device)
    t = lower + (upper - lower) * u  # [batch_size, nb_bins]
    delta = torch.cat((t[:, 1:] - t[:, :-1], torch.tensor([1e10], device=device).expand(ray_origins.shape[0], 1)), -1)

    x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1)  # [batch_size, nb_bins, 3]
    ray_directions = ray_directions.expand(nb_bins, ray_directions.shape[0], 3).transpose(0, 1)

    colors, sigma = nerf_model(x.reshape(-1, 3), ray_directions.reshape(-1, 3))
    colors = colors.reshape(x.shape)
    sigma = sigma.reshape(x.shape[:-1])

    alpha = 1 - torch.exp(-sigma * delta)  # [batch_size, nb_bins]
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)
    c = (weights * colors).sum(dim=1)  # Pixel values
    weight_sum = weights.sum(-1).sum(-1)  # Regularization for white background
    return c + 1 - weight_sum.unsqueeze(-1)


def train(nerf_model, optimizer, scheduler, data_loader, device='cpu', hn=0, hf=1, nb_epochs=int(1e5), nb_bins=192, H=400, W=400):
    training_loss = []
    for _ in (range(nb_epochs)):
        for ep, batch in enumerate(tqdm(data_loader)):
            ray_origins = batch[:, :3].to(device)
            ray_directions = batch[:, 3:6].to(device)
            ground_truth_px_values = batch[:, 6:].to(device)

            regenerated_px_values = render_rays(nerf_model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins)
            loss = ((ground_truth_px_values - regenerated_px_values) ** 2).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss.append(loss.item())
        scheduler.step()
    torch.save(model.state_dict(), 'final_model.pth')
    return training_loss


if __name__ == '__main__':
    device = 'cuda'
    training_dataset = torch.from_numpy(np.load('data/smallcity/training_data.pkl', allow_pickle=True))
    testing_dataset = torch.from_numpy(np.load('data/smallcity/training_data.pkl', allow_pickle=True))
    model = NerfModel(hidden_dim=256).to(device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=[2, 4, 8], gamma=0.5)

    data_loader = DataLoader(training_dataset, batch_size=1024, shuffle=True)
    train(model, model_optimizer, scheduler, data_loader, nb_epochs=1, device=device, hn=2, hf=6, nb_bins=192, H=1080,
          W=1920)
    for img_index in range(0, 10):
        test(2, 6, testing_dataset, img_index=img_index, nb_bins=192, H=1080, W=1920)
