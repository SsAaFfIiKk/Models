import cv2
import torch
import librosa
import numpy as np
from math import ceil
from torch.autograd import Variable
from torchvision.transforms import transforms


def chunkizer(audio, chunk_length, sr=48000, trim_threshold=2):
    num_chunks = ceil(librosa.get_duration(audio, sr=sr) / chunk_length)
    chunks = []
    for i in range(num_chunks):
        chunks.append(audio[i*chunk_length*sr:(i+1)*chunk_length*sr])
    if trim_threshold:
        if len(chunks[-1]) <= trim_threshold*sr:
            chunks = chunks[:-1]
    return chunks


def load_model(model, path_to_weights, device):
    model.load_state_dict(torch.load(path_to_weights))
    model.to(device)
    model.eval()
    return model


def zero_pad(spec):
    rows, cols = spec.shape
    desired_shape = rows
    spec = scale_minmax(spec, 0, 255).astype(np.uint8)
    zeros = np.zeros((rows, desired_shape), dtype=np.uint8)
    np.random.seed(42)
    beginning_col = np.random.randint(0, desired_shape - cols + 1)
    zeros[..., beginning_col:beginning_col + cols] = spec
    spec_z = zeros
    return spec_z


def scale_minmax(x, mn=0.0, mx=1.0):
    x_std = (x - x.min()) / (x.max() - x.min())
    x_scaled = x_std * (mx - mn) + mn
    return x_scaled


def get_logits(audio, model, device):
    img_shape = 224
    spec = librosa.feature.melspectrogram(
        y=audio, sr=48000,
        n_mels=256 + 32, n_fft=2048, hop_length=512)
    db = librosa.power_to_db(spec)
    spec = zero_pad(db)
    fliped_img = np.flip(spec, axis=0)
    resized_img = cv2.resize(fliped_img, dsize=(img_shape, img_shape), interpolation=cv2.INTER_CUBIC)
    t_img = transform(resized_img)
    u_img = t_img.unsqueeze(0)
    v_img = Variable(u_img)
    v_img = v_img.to(device)
    output_logits = model(v_img).detach().numpy()[0].tolist()
    return output_logits


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.225])])
