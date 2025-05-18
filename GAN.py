import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
# ハイパーパラメータ
latent_dim = 100  # ランダムノイズの次元(数字の複雑さ)
img_shape = (1, 28, 28) # (色, 高さ, 幅)
img_size = 28 * 28     # 画像の総ピクセル数
batch_size = 64
epochs = 50 # 学習エポック数 (訓練の回数)
lr = 0.0002     # 学習率

# デバイス設定 (GPUを使用)
device = torch.device("cuda")
# データの前処理(画像の大きさをコンピュータの扱えるものにする)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]) 
])

# 学習元のデータを用意
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)# データをまとめて訓練に使う
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),# 具体的な形にする
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, img_size),
            nn.Tanh() # 出力を-1から1の範囲に
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape) # (バッチサイズ, 1, 28, 28) の形状に戻す
        return img
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid() # 出力を0から1の範囲に (本物である確率)
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1) # 画像をフラットにする
        validity = self.model(img_flat)
        return validity
# モデルのインスタンス化
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# 損失関数 
adversarial_loss = nn.BCELoss().to(device)

# オプティマイザ 
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
# 生成画像を保存するための関数 
def save_generated_images(epoch, generator, latent_dim, device, n_images=25):
    z = torch.randn(n_images, latent_dim).to(device)
    gen_imgs = generator(z).detach().cpu()
    gen_imgs = 0.5 * gen_imgs + 0.5 # -1~1 を 0~1 に戻す
    fig, axs = plt.subplots(5, 5, figsize=(5,5))
    cnt = 0
    for i in range(5):
        for j in range(5):
            axs[i,j].imshow(gen_imgs[cnt, 0, :, :], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig(f"gan_images/mnist_generated_epoch_{epoch}.png") # gan_imagesフォルダの作成
    plt.close()

# 学習ループ
output_dir = "gan_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
for epoch in range(epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # 本物の画像と偽物の画像にラベリング
        real_label = torch.ones(imgs.size(0), 1).to(device)  # 本物は1
        fake_label = torch.zeros(imgs.size(0), 1).to(device) # 偽物は0

        #  Discriminatorの学習
    
        optimizer_D.zero_grad()

        # 本物の画像での損失
        real_imgs = imgs.to(device)
        real_loss = adversarial_loss(discriminator(real_imgs), real_label)

        # 偽物の画像での損失
        z = torch.randn(imgs.size(0), latent_dim).to(device) # ランダムノイズ生成
        fake_imgs = generator(z) # Generatorで偽画像を生成
        fake_loss = adversarial_loss(discriminator(fake_imgs.detach()), fake_label) # Generatorの勾配は計算しない
        

        # Discriminatorの合計損失
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        #  Generatorの学習
        optimizer_G.zero_grad()

        # Generatorが生成した偽画像を、Discriminatorが本物と誤認するように学習
        gen_imgs = generator(z) # 新たに偽画像を生成 (上のfake_imgsとは異なる)
        g_loss = adversarial_loss(discriminator(gen_imgs), real_label) # Discriminatorに騙させる

        g_loss.backward()
        optimizer_G.step()

        # ログ出力
        if (i + 1) % 200 == 0:
            print(
                f"[Epoch {epoch+1}/{epochs}] [Batch {i+1}/{len(dataloader)}] "
                f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]"
            )

    # 各エポック終了時に生成画像を保存 (オプション)
    if (epoch + 1) % 5 == 0: # 5エポックごとに保存
        save_generated_images(epoch + 1, generator, latent_dim, device)
        print(f"Epoch {epoch+1}: Generated images saved.")

print("学習完了！")
# 学習済みのGeneratorを使って画像を生成
def show_generated_images(generator, latent_dim, device, n_images=25):
    z = torch.randn(n_images, latent_dim).to(device)
    gen_imgs = generator(z).detach().cpu()
    gen_imgs = 0.5 * gen_imgs + 0.5 # -1~1 を 0~1 に戻す
    fig, axs = plt.subplots(5, 5, figsize=(5,5))
    cnt = 0
    for i in range(5):
        for j in range(5):
            axs[i,j].imshow(gen_imgs[cnt, 0, :, :], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    plt.show()
