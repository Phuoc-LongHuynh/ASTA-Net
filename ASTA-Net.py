import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy
import copy
from torch.optim.lr_scheduler import ReduceLROnPlateau

class SemanticSegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_paths = sorted([os.path.join(image_dir, img) for img in os.listdir(image_dir)])
        self.label_paths = sorted([os.path.join(label_dir, lbl) for lbl in os.listdir(label_dir)])
        self.class_colors = {
            (255, 255, 255): 0, 
            (160, 160, 160): 1, 
            (80, 80, 80): 2,    
            (0, 0, 0): 3        
        }
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = cv2.imread(self.label_paths[idx])
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

        label_mask = np.zeros(label.shape[:2], dtype=np.uint8)
        for rgb, idx in self.class_colors.items():
            label_mask[np.all(label == rgb, axis=-1)] = idx

        if self.transform:
            image = self.transform(image)
            label_mask = torch.from_numpy(label_mask).long()
 
        return image, label_mask

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])

train_dataset = SemanticSegmentationDataset(
    image_dir='/kaggle/input/5g-lte-nr-j03/J03_spectrumm/train/data',
    label_dir='/kaggle/input/5g-lte-nr-j03/J03_spectrumm/train/label',
    transform=train_transform)

val_dataset = SemanticSegmentationDataset(
    image_dir='/kaggle/input/5g-lte-nr-j03/J03_spectrumm/test/data',
    label_dir='/kaggle/input/5g-lte-nr-j03/J03_spectrumm/test/label',
    transform=train_transform)


def train_epoch(model, dataloader, main_loss_fn, aux_loss_fn, main_loss, lambda_aux, optimizer, device, num_classes):
    model.train() 
    running_total_loss, running_main_loss, running_aux_loss = 0.0, 0.0, 0.0

    accuracy_metric = MulticlassAccuracy(num_classes=num_classes).to(device)
    iou_metric = MulticlassJaccardIndex(num_classes=num_classes).to(device)
    
    pbar = tqdm(dataloader, desc='Training', unit='batch')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        final_pred, aux_output = model(images)
        loss_main = main_loss_fn(final_pred, labels)
        gt_downsampled = F.interpolate(
            labels.unsqueeze(1).float(),
            size=aux_output.shape[2:],
            mode='nearest'
        ).squeeze(1).long()
        loss_aux = aux_loss_fn(aux_output, gt_downsampled)
        total_loss = main_loss * loss_main + lambda_aux * loss_aux
        total_loss.backward()
        optimizer.step()
        running_total_loss += total_loss.item() * images.size(0)
        running_main_loss += loss_main.item() * images.size(0)
        running_aux_loss += loss_aux.item() * images.size(0)
        preds = torch.argmax(final_pred, dim=1)
        accuracy_metric.update(preds, labels)
        iou_metric.update(preds, labels)
        pbar.set_postfix({
            'Total Loss': f'{total_loss.item():.4f}',
            'mIoU': f'{iou_metric.compute():.4f}',
        })

    epoch_total_loss = running_total_loss / len(dataloader.dataset)
    epoch_main_loss = running_main_loss / len(dataloader.dataset)
    epoch_aux_loss = running_aux_loss / len(dataloader.dataset)
    mean_accuracy = accuracy_metric.compute().cpu().item()
    mean_iou = iou_metric.compute().cpu().item()
    
    return {
        "total_loss": epoch_total_loss,
        "main_loss": epoch_main_loss,
        "aux_loss": epoch_aux_loss,
        "accuracy": mean_accuracy,
        "iou": mean_iou
    }

def evaluate(model, dataloader, main_loss_fn, device, num_classes):
    model.eval()
    running_loss = 0.0
    

    accuracy_metric = MulticlassAccuracy(num_classes=num_classes).to(device)
    iou_metric = MulticlassJaccardIndex(num_classes=num_classes).to(device)    
    pbar = tqdm(dataloader, desc='Evaluating', unit='batch')
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = main_loss_fn(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            accuracy_metric.update(preds, labels)
            iou_metric.update(preds, labels)
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'mIoU': f'{iou_metric.compute():.4f}',
            })
            
    epoch_loss = running_loss / len(dataloader.dataset)
    mean_accuracy = accuracy_metric.compute().cpu().item()
    mean_iou = iou_metric.compute().cpu().item()
    
    return {
        "loss": epoch_loss,
        "accuracy": mean_accuracy,
        "iou": mean_iou
    }


class ASA(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.q_proj = nn.Linear(in_channels, in_channels)
        self.k_proj = nn.Linear(in_channels, in_channels)
        self.w_a = nn.Parameter(torch.randn(1, 1, in_channels))
        self.final_linear = nn.Linear(in_channels, in_channels)
        self.norm = nn.LayerNorm(in_channels)
        self.relu = nn.GELU()

    def forward(self, x):
        q_seq = self.q_proj(x)
        k_seq = self.k_proj(x)
        scores = (q_seq * self.w_a).sum(dim=2)
        weighted_q = scores.unsqueeze(-1) * q_seq
        D = q_seq.shape[-1]
        q_pre_softmax = torch.sum(weighted_q, dim=1, keepdim=True) / (D ** 0.5)
        feature_attention_weights = torch.softmax(q_pre_softmax, dim=2)
        out = k_seq * feature_attention_weights
        out = self.final_linear(self.relu(out))
        out = out + self.norm(q_seq)
        return out

class MixFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, H, W):
        x = self.fc1(x)
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.fc2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., sr_ratio=1., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = ASA(in_channels=dim)
        self.norm2 = norm_layer(dim)
        self.mlp = MixFFN(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x, H, W):
        x = self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x), H, W)
        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=768, patch_size=7, stride=4):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=(patch_size // 2, patch_size // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class Backbone(nn.Module):
    def __init__(self, in_chans=3):
        super().__init__()
        embed_dims = [32, 64, 160, 256]
        mlp_ratios = [4, 4, 4, 4]
        depths = [2, 2, 2, 2]
        sr_ratios = [8, 4, 2, 1]
        self.embed_dims = embed_dims

        self.patch_embed1 = OverlapPatchEmbed(in_chans=in_chans, embed_dim=embed_dims[0], patch_size=7, stride=4)
        self.block1 = nn.ModuleList([TransformerBlock(dim=embed_dims[0], mlp_ratio=mlp_ratios[0], sr_ratio=sr_ratios[0]) for _ in range(depths[0])])
        self.norm1 = nn.LayerNorm(embed_dims[0])

        self.patch_embed2 = OverlapPatchEmbed(in_chans=embed_dims[0], embed_dim=embed_dims[1], patch_size=3, stride=2)
        self.block2 = nn.ModuleList([TransformerBlock(dim=embed_dims[1], mlp_ratio=mlp_ratios[1], sr_ratio=sr_ratios[1]) for _ in range(depths[1])])
        self.norm2 = nn.LayerNorm(embed_dims[1])

        self.patch_embed3 = OverlapPatchEmbed(in_chans=embed_dims[1], embed_dim=embed_dims[2], patch_size=3, stride=2)
        self.block3 = nn.ModuleList([TransformerBlock(dim=embed_dims[2], mlp_ratio=mlp_ratios[2], sr_ratio=sr_ratios[2]) for _ in range(depths[2])])
        self.norm3 = nn.LayerNorm(embed_dims[2])

        self.patch_embed4 = OverlapPatchEmbed(in_chans=embed_dims[2], embed_dim=embed_dims[3], patch_size=3, stride=2)
        self.block4 = nn.ModuleList([TransformerBlock(dim=embed_dims[3], mlp_ratio=mlp_ratios[3], sr_ratio=sr_ratios[3]) for _ in range(depths[3])])
        self.norm4 = nn.LayerNorm(embed_dims[3])

        self.output_conv2 = nn.Sequential(
            nn.Conv2d(embed_dims[1], embed_dims[1], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims[1]),
            nn.GELU()
        )
        self.output_conv3 = nn.Sequential(
            nn.Conv2d(embed_dims[2], embed_dims[2], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims[2]),
            nn.GELU()
        )
        self.output_conv4 = nn.Sequential(
            nn.Conv2d(embed_dims[3], embed_dims[3], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims[3]),
            nn.GELU()
        )

        self.upsample_s2 = nn.ConvTranspose2d(embed_dims[1], embed_dims[1], kernel_size=2, stride=2)
        self.upsample_s3 = nn.ConvTranspose2d(embed_dims[2], embed_dims[2], kernel_size=4, stride=4)
        self.upsample_s4 = nn.ConvTranspose2d(embed_dims[3], embed_dims[3], kernel_size=8, stride=8)

    def forward(self, x):
        B = x.shape[0]
        x, H, W = self.patch_embed1(x)
        for blk in self.block1: x = blk(x, H, W)
        s1_raw = self.norm1(x).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x, H, W = self.patch_embed2(s1_raw)
        for blk in self.block2: x = blk(x, H, W)
        s2_raw = self.norm2(x).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x, H, W = self.patch_embed3(s2_raw)
        for blk in self.block3: x = blk(x, H, W)
        s3_raw = self.norm3(x).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x, H, W = self.patch_embed4(s3_raw)
        for blk in self.block4: x = blk(x, H, W)
        s4_raw = self.norm4(x).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        s2_out = self.output_conv2(s2_raw)
        s3_out = self.output_conv3(s3_raw)
        s4_out = self.output_conv4(s4_raw)

        s2_up = self.upsample_s2(s2_out)
        s3_up = self.upsample_s3(s3_out)
        s4_up = self.upsample_s4(s4_out)
        return [s2_up, s3_up, s4_up]

class SRM(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        if out_channels % 2 != 0:
            out_channels += 1
        self.out_channels = out_channels
        mlp_hidden_dim = out_channels
        self.project_x = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm_x = nn.BatchNorm2d(out_channels)
        self.to_gamma_spatial_mlp = nn.Sequential(
            nn.Conv2d(out_channels, mlp_hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(mlp_hidden_dim, out_channels, 1), nn.GELU(),
        )
        self.to_beta_spatial_mlp = nn.Sequential(
            nn.Conv2d(out_channels, mlp_hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(mlp_hidden_dim, out_channels, 1), nn.GELU(),
        )
        self.relu = nn.GELU()

    def forward(self, x, x_res):
        x_proj = self.norm_x(self.project_x(x))
        gamma_spatial = self.to_gamma_spatial_mlp(x_res)
        beta_spatial = self.to_beta_spatial_mlp(x_res)
        x_half1 = x_proj[:, :self.out_channels // 2, :, :]
        x_half2 = x_proj[:, self.out_channels // 2:, :, :]
        feat_sin = torch.sin(x_half1)
        feat_cos = torch.cos(x_half2)
        x_activated = torch.cat((feat_sin, feat_cos), dim=1)
        out = x_activated * gamma_spatial + beta_spatial
        out = out + x_proj
        return self.relu(out)

class LCM(nn.Module):
    def __init__(self, in_channels, rate=1, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels * 2
        self.out_channels = out_channels
        self.rate = rate
        reduced = out_channels // 2
        self.conv3x3 = nn.Conv2d(in_channels, reduced, kernel_size=3, padding=1, bias=False)
        self.bnconv3x3 = nn.BatchNorm2d(reduced)
        self.acconv3x3 = nn.Conv2d(reduced, reduced, kernel_size=3, padding=rate, dilation=rate, bias=False)
        self.bnacconv3x3 = nn.BatchNorm2d(reduced)
        self.relu = nn.GELU()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, x_res):
        x_1 = self.relu(self.bnconv3x3(self.conv3x3(x)))
        x_2 = self.relu(self.bnacconv3x3(self.acconv3x3(x_1)))
        x_combined = torch.cat((x_1, x_2), dim=1)
        return x_combined + x_res

class SynergicFusion(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, num_classes, kernel_size=3, padding=1, bias=False),
        )
        self.upsample_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )
        self.upsample_x4 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=4, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, b1, b2, b3):
        g = self.sigmoid(b1 + b3)
        b1_attended = b1 * g
        b2_attended = b2 * (1 - g)
        fused_output = torch.cat([b1_attended, b2_attended], dim=1)
        out = self.final_conv(self.upsample_x4(self.upsample_conv(fused_output)))
        return out

class myModel(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super().__init__()
        self.backbone = Backbone(in_chans=in_channels)
        self.first_layer_1 = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Conv2d(48, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.first_layer_2 = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Conv2d(48, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.SRM_0 = SRM(in_channels=32, out_channels=64)
        self.SRM_1 = SRM(in_channels=64, out_channels=160)
        self.SRM_2 = SRM(in_channels=160, out_channels=256)
        self.LCM_0 = LCM(in_channels=32, out_channels=64, rate=1)
        self.LCM_1 = LCM(in_channels=64, out_channels=160, rate=2)
        self.LCM_2 = LCM(in_channels=160, out_channels=256, rate=3)
        self.SynergicFusion = SynergicFusion(in_channels=256, num_classes=num_classes)        
        self.S_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, padding=0),
        )

    def forward(self, x):
        # Transformer encoder
        s1, s2, s3 = self.backbone(x)
        # SR path
        x1 = self.first_layer_1(x)
        b1_0 = self.SRM_0(x1, s1)
        b1_1 = self.SRM_1(b1_0, s2)
        b1_2 = self.SRM_2(b1_1, s3)
        # PS path
        x2 = self.first_layer_2(x)
        b2_0 = self.LCM_0(x2, s1)
        b2_1 = self.LCM_1(b2_0, s2)
        b2_2 = self.LCM_2(b2_1, s3)
        out = self.SynergicFusion(b1_2, b2_2, s3)
        if self.training:
            return out, self.S_head(b2_2)
        else:
            return out
        
# --- AGGREGATION LOSS ---
class AGGREGATIONLOSS(nn.Module):
    def __init__(self, num_classes, temperature=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature

    def forward(self, features, gt_mask):
        B, C, H, W = features.shape
        N = B * H * W
        F_flat = features.permute(0, 2, 3, 1).reshape(N, C)
        y_flat = gt_mask.view(N)
        mask = F.one_hot(y_flat, num_classes=self.num_classes).float()
        class_counts = mask.sum(dim=0) + 1e-8
        prototypes = (mask.T @ F_flat) / class_counts.unsqueeze(1)
        prototypes = F.normalize(prototypes, dim=1)
        F_flat = F.normalize(F_flat, dim=1)
        logits = (F_flat @ prototypes.T) / self.temperature
        loss = F.cross_entropy(logits, y_flat, reduction='mean')
        return loss
    

train_dataloader = DataLoader(train_dataset, batch_size=32, pin_memory=True, shuffle=True, num_workers=os.cpu_count())
val_dataloader = DataLoader(val_dataset, batch_size=32, pin_memory=True, shuffle=False, num_workers=os.cpu_count())
print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = 4
model = myModel(classes)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
total_params = count_parameters(model)
print(f"Total parameters: {total_params}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = len(classes) if isinstance(classes, list) else classes
model = myModel(num_classes)
model = model.to(device)
model = nn.DataParallel(model)

main_loss_fn = nn.CrossEntropyLoss().to(device)
aux_loss_fn = AGGREGATIONLOSS(num_classes=num_classes).to(device)
lambda_aux = 0.5
main_loss = 0.7
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

num_epochs = 60
epoch_saved = 0
best_mIoU_val = 0.0
best_model_state = None

train_losses = []
val_losses = []
train_mAccs = []
val_mAccs = []
train_mIoUs = []
val_mIoUs = []

# --- TRAINING LOOP ---
for epoch in range(num_epochs):
    train_metrics = train_epoch(model, train_dataloader, main_loss_fn, aux_loss_fn, main_loss, lambda_aux, optimizer, device, num_classes)
    val_metrics = evaluate(model, val_dataloader, main_loss_fn, device, num_classes)

    epoch_loss_train = train_metrics["total_loss"]
    main_loss_train = train_metrics["main_loss"]
    aux_loss_train = train_metrics["aux_loss"]
    mAcc_train = train_metrics["accuracy"]
    mIoU_train = train_metrics["iou"]

    epoch_loss_val = val_metrics["loss"]
    mAcc_val = val_metrics["accuracy"]
    mIoU_val = val_metrics["iou"]

    train_losses.append(epoch_loss_train)
    val_losses.append(epoch_loss_val)
    train_mAccs.append(mAcc_train)
    val_mAccs.append(mAcc_val)
    train_mIoUs.append(mIoU_train)
    val_mIoUs.append(mIoU_val)

    current_lr = optimizer.param_groups[0]['lr']
    print(f"--- Epoch {epoch + 1}/{num_epochs} ---")
    print(f"Learning Rate: {current_lr:.6f}")

    print(f"Train: Total Loss: {epoch_loss_train:.4f} (Main: {main_loss_train:.4f}, Aux: {aux_loss_train:.4f}) | mAcc: {mAcc_train:.4f} | mIoU: {mIoU_train:.4f}")
    print(f"Val:   Loss: {epoch_loss_val:.4f} | mAcc: {mAcc_val:.4f} | mIoU: {mIoU_val:.4f}")

    scheduler.step(epoch_loss_val)

    if mIoU_val >= best_mIoU_val:
        epoch_saved = epoch + 1
        best_mIoU_val = mIoU_val
        model_to_save_state_dict = model.module if isinstance(model, nn.DataParallel) else model
        best_model_state = copy.deepcopy(model_to_save_state_dict.state_dict())
        print(f"*** Best model state saved at epoch {epoch_saved} with mIoU: {best_mIoU_val:.4f} ***\n")

print("===================")
print(f"Best Model at epoch: {epoch_saved}")

if best_model_state is not None:
    best_model_path_state_dict = f"/kaggle/working/best_model_epoch_{epoch_saved}_mIoU_{best_mIoU_val:.4f}_statedict.pth"
    torch.save(best_model_state, best_model_path_state_dict)
    print(f"Saved best model state_dict at: {best_model_path_state_dict}")
else:
    print("No better model found during this training session to save state_dict.")

print(f"Saving state_dict of the model from the last epoch {num_epochs}...")
model_to_save_final_state_dict = model.module if isinstance(model, nn.DataParallel) else model
final_model_path_state_dict = f"/kaggle/working/final_epoch_{num_epochs}_statedict.pth"
torch.save(model_to_save_final_state_dict.state_dict(), final_model_path_state_dict)
print(f"Saved final model state_dict at: {final_model_path_state_dict}")
