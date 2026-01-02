import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import random
from IPython.display import HTML, display
from base64 import b64encode
import torchvision.models as models
from torch.cuda.amp import autocast, GradScaler

ARCHIVE_URL = "https://archive.org/download/TouhouBadApple/Touhou%20-%20Bad%20Apple.mp4"
FPS = 12
RES = 128
CHUNK_SIZE = 16
EPOCHS = 2000
AUDIO_SR = 22050
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"--- ЗАПУСК PROJECT-86: BAD AI (FIXED) на {DEVICE} ---")


print("--- Шаг 1: Скачивание и обработка ---")

if not os.path.exists("source.mp4"):
    if not os.path.exists("raw_full.mp4"):
        os.system(f'wget -O raw_full.mp4 "{ARCHIVE_URL}"')

    if os.path.getsize("raw_full.mp4") < 1000:
        raise FileNotFoundError("Файл raw_full.mp4 не скачался!")

    os.system(f'ffmpeg -y -i raw_full.mp4 -vf "scale={RES}:{RES},fps={FPS},format=gray" -an -c:v libx264 source.mp4 -loglevel error')
    os.system(f'ffmpeg -y -i raw_full.mp4 -ac 1 -ar {AUDIO_SR} -vn source.wav -loglevel error')

def load_data():
    cap = cv2.VideoCapture("source.mp4")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame[:,:,0] / 255.0)
    cap.release()
    v_tensor = torch.tensor(np.array(frames), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    waveform, sr = torchaudio.load("source.wav")
    return v_tensor, waveform

print("Загрузка данных в RAM...")
video_cpu, audio_cpu = load_data()

TOTAL_FRAMES = video_cpu.shape[2]
TOTAL_SAMPLES = audio_cpu.shape[1]
print(f"✅ Данные в памяти. Кадров: {TOTAL_FRAMES}")


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features

        
        self.slice1 = vgg[:4].eval()
        self.slice2 = vgg[4:9].eval()

        for p in self.parameters(): p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, input, target):
        if input.shape[2] == 0: return 0
        b, c, t, h, w = input.shape
        
        in_flat = input.permute(0, 2, 1, 3, 4).reshape(-1, 1, h, w).repeat(1, 3, 1, 1)
        tg_flat = target.permute(0, 2, 1, 3, 4).reshape(-1, 1, h, w).repeat(1, 3, 1, 1)

        
        in_flat = (in_flat - self.mean) / self.std
        tg_flat = (tg_flat - self.mean) / self.std

        loss = 0

        
        x1 = self.slice1(in_flat)
        y1 = self.slice1(tg_flat)
        loss += F.l1_loss(x1, y1)

        
        x2 = self.slice2(x1)
        y2 = self.slice2(y1)
        loss += F.l1_loss(x2, y2)

        return loss


class Swish(nn.Module):
    def forward(self, x): return x * torch.sigmoid(x)

class LinearAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 4, 1).reshape(B, -1, C)
        q, k, v = self.q(x_flat), self.k(x_flat), self.v(x_flat)
        q, k = F.silu(q), F.silu(k)
        context = (k * v).mean(dim=1, keepdim=True)
        out = (q * context) + v
        out = self.proj(out)
        return out.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)

class LVG_Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(4, dim)
        self.attn = LinearAttention(dim)
        self.norm2 = nn.GroupNorm(4, dim)
        self.conv = nn.Conv3d(dim, dim, 3, padding=1)
        self.act = Swish()

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.act(self.conv(self.norm2(x)))
        return x

class FullGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.video_latent = nn.Parameter(torch.randn(1, 32, TOTAL_FRAMES // 4 + 8, RES // 4, RES // 4) * 0.1)
        self.audio_latent = nn.Parameter(torch.randn(1, TOTAL_SAMPLES) * 0.01)

        self.video_net = nn.Sequential(
            LVG_Block(32),
            LVG_Block(32),
            nn.ConvTranspose3d(32, 16, (4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            Swish(),
            nn.ConvTranspose3d(16, 8, (4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            Swish(),
            nn.Conv3d(8, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward_video(self, start_frame, end_frame):
        l_start = start_frame // 4
        l_end = end_frame // 4 + 1
        l_max = self.video_latent.shape[2]
        l_end = min(l_end, l_max)

        chunk = self.video_latent[:, :, l_start:l_end, :, :]
        out = self.video_net(chunk)

        valid_len = end_frame - start_frame
        if out.shape[2] > valid_len:
            out = out[:, :, :valid_len, :, :]
        return out

    def forward_audio(self):
        return torch.tanh(self.audio_latent)


model = FullGenerator().to(DEVICE)
vgg_loss = PerceptualLoss().to(DEVICE)
mse = nn.MSELoss()
l1 = nn.L1Loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
scaler = GradScaler()

print(f"--- START TRAINING ({EPOCHS} steps) ---")

for step in range(EPOCHS + 1):
    optimizer.zero_grad()

    
    start_t = random.randint(0, TOTAL_FRAMES - CHUNK_SIZE - 1)
    end_t = start_t + CHUNK_SIZE
    target_chunk = video_cpu[:, :, start_t:end_t, :, :].to(DEVICE)

    with autocast():
        pred_chunk = model.forward_video(start_t, end_t)

        if pred_chunk.shape[2] != target_chunk.shape[2]:
            min_len = min(pred_chunk.shape[2], target_chunk.shape[2])
            pred_chunk = pred_chunk[:, :, :min_len]
            target_chunk = target_chunk[:, :, :min_len]

        loss_pixel = mse(pred_chunk, target_chunk)
        
        loss_vgg = vgg_loss(pred_chunk, target_chunk) * 0.05

        
        a_start = int((start_t / FPS) * AUDIO_SR)
        a_end = int((end_t / FPS) * AUDIO_SR)
        a_start = max(0, min(a_start, TOTAL_SAMPLES - 100))
        a_end = min(TOTAL_SAMPLES, a_end)

        loss_audio = torch.tensor(0.0, device=DEVICE)
        if a_end > a_start:
            pred_audio = model.forward_audio()[:, a_start:a_end]
            target_audio = audio_cpu[:, a_start:a_end].to(DEVICE)
            loss_audio = l1(pred_audio, target_audio) * 0.5

        total_loss = loss_pixel + loss_vgg + loss_audio

    scaler.scale(total_loss).backward()
    scaler.step(optimizer)
    scaler.update()

    if step % 50 == 0:
        print(f"Step {step} | Loss: {total_loss.item():.4f} (Vid: {loss_pixel.item():.4f} | Aud: {loss_audio.item():.4f})")

    if step % 500 == 0 and step > 0:
        torch.save(model.state_dict(), f"Project-86-badAI_ckpt_{step}.pt")

FINAL_WEIGHTS = "Project-86-badAI.pt"
torch.save(model.state_dict(), FINAL_WEIGHTS)
print(f"✅ Training Done. Weights saved: {FINAL_WEIGHTS}")


print("--- RENDERING FULL VIDEO ---")
model.eval()
torch.cuda.empty_cache()

full_video_frames = []
BATCH_RENDER = 200

with torch.no_grad():
    for t in range(0, TOTAL_FRAMES, BATCH_RENDER):
        end_t = min(t + BATCH_RENDER, TOTAL_FRAMES)
        chunk = model.forward_video(t, end_t).cpu()
        chunk = chunk.squeeze(0).squeeze(0).numpy()
        chunk = (np.clip(chunk, 0, 1) * 255).astype(np.uint8)
        full_video_frames.append(chunk)
        print(f"Rendering: {t}/{TOTAL_FRAMES}")

    full_audio = model.forward_audio().cpu().squeeze().numpy()

video_data = np.concatenate(full_video_frames, axis=0)
v_out = cv2.VideoWriter('gen_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), FPS, (RES, RES), False)
for frame in video_data:
    v_out.write(frame)
v_out.release()

torchaudio.save("gen_audio.wav", torch.tensor(full_audio).unsqueeze(0), AUDIO_SR)

os.system("ffmpeg -y -i gen_video.mp4 -i gen_audio.wav -c:v libx264 -c:a aac -shortest Project-86-Result.mp4 -loglevel quiet")

def show_final():
    mp4 = open('Project-86-Result.mp4','rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    return f"""
    <h3>PROJECT-86: BAD APPLE COMPLETE</h3>
    <video width=512 height=512 controls>
          <source src="{data_url}" type="video/mp4">
    </video>
    """

display(HTML(show_final()))
