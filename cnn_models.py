import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Single patch-position CNN autoencoder (CNN-PatchAE branch)
class PatchNet(nn.Module):
    def __init__(self, channel, feat_map):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channel,    feat_map,   kernel_size=3, stride=1), nn.ReLU(),
            nn.Conv2d(feat_map,   feat_map*2, kernel_size=3, stride=1), nn.ReLU(),
            nn.Conv2d(feat_map*2, feat_map*4, kernel_size=3, stride=1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feat_map*4, feat_map*2, kernel_size=3, stride=1), nn.ReLU(),
            nn.ConvTranspose2d(feat_map*2, feat_map,   kernel_size=3, stride=1), nn.ReLU(),
            nn.ConvTranspose2d(feat_map,   channel,    kernel_size=3, stride=1), nn.Sigmoid(),
        )

    def forward(self, x):
        features      = self.encoder(x)
        reconstructed = self.decoder(features)
        return reconstructed, features


# Collection of P independent PatchNet branches, one per patch position (CNN-PatchAE)
class PatchNetMulti(nn.Module):
    def __init__(self, channel: int, feat_map: int, n_parts=4, exp=5):
        super().__init__()

        if isinstance(n_parts, (tuple, list)):
            self.rows, self.cols = int(n_parts[0]), int(n_parts[1])
        else:
            n_parts = int(n_parts)
            sq = math.isqrt(n_parts)
            if sq * sq != n_parts:
                raise ValueError(f"n_parts={n_parts} is not a perfect square.")
            self.rows, self.cols = sq, sq

        self.n_parts     = self.rows * self.cols
        self.n_parts_arg = n_parts
        self.channel     = channel
        self.feat_map    = feat_map

        for idx in range(self.n_parts):
            setattr(self, f"cnn_p{n_parts}_side{idx}", PatchNet(channel, feat_map))

    def _extract_patches(self, x):
        B, C, H, W = x.shape
        ph = H // self.rows
        pw = W // self.cols
        patches = []
        for r in range(self.rows):
            for c in range(self.cols):
                patches.append(x[:, :, r*ph:(r+1)*ph, c*pw:(c+1)*pw])
        return patches

    def forward(self, x):
        patches = self._extract_patches(x)
        recons, feats = [], []
        for idx in range(self.n_parts):
            net = getattr(self, f"cnn_p{self.n_parts_arg}_side{idx}")
            recon, feat = net(patches[idx])
            recons.append(recon)
            feats.append(feat)
        return recons, feats

    @property
    def branch_names(self):
        return [f"cnn_p{self.n_parts_arg}_side{i}" for i in range(self.n_parts)]


# Global Feature Centroid (GFC): per-depth running-average spatial maps for cross-task regularization
class GFC(nn.Module):
    def __init__(self, momentum: float = 0.9):
        super().__init__()
        self.momentum = momentum
        self.maps = nn.ParameterList()
        self.register_buffer('is_initialised', torch.tensor(False))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Rebuild ParameterList from checkpoint
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
        map_keys = sorted(
            [k for k in state_dict.keys() if k.startswith(prefix + 'maps.')],
            key=lambda x: int(x.split('.')[-1])
        )
        if map_keys:
            self.maps = nn.ParameterList()
            for key in map_keys:
                param = nn.Parameter(state_dict[key].float())
                self.maps.append(param)
            self.is_initialised.fill_(True)
            print(f"  [GFC-CNN] Loaded {len(map_keys)} GFC maps from checkpoint.")

    @torch.no_grad()
    def initialise(self, frozen_layers: list):
        # Task 1: initialize one GFC centroid per depth from frozen conv outputs
        means = [fl.detach().mean(dim=0, keepdim=True) for fl in frozen_layers]
        self.maps = nn.ParameterList([nn.Parameter(m.clone()) for m in means])
        self.is_initialised.fill_(True)
        print("  [GFC-CNN] Initialised from task-1 frozen conv outputs.")

    @torch.no_grad()
    def update(self, adapter_layers: list):
        # Tasks > 1: EMA update of each depth centroid from adapter outputs
        if len(self.maps) == 0:
            if not self.is_initialised.item():
                raise RuntimeError("GFC not initialised. Call initialise() first.")
            else:
                raise RuntimeError("GFC marked as initialised but maps are empty.")
        for gfc_map, al in zip(self.maps, adapter_layers):
            new_mean = al.detach().mean(dim=0, keepdim=True).float().to(gfc_map.device)
            gfc_map.data.mul_(self.momentum).add_(new_mean * (1.0 - self.momentum))
        print(f"  [GFC-CNN] Updated from adapter outputs (momentum={self.momentum}).")

    def alignment_loss(self, adapter_layers: list) -> torch.Tensor:
        # Drift loss: MSE between adapter outputs and detached GFC maps, averaged across depths
        if len(self.maps) == 0:
            return torch.tensor(0.0, device=adapter_layers[0].device)
        loss = torch.tensor(0.0, device=adapter_layers[0].device)
        for al, gfc_map in zip(adapter_layers, self.maps):
            B  = al.shape[0]
            pt = gfc_map.detach().expand(B, -1, -1, -1)
            loss = loss + F.mse_loss(al, pt.type_as(al))
        return loss / len(adapter_layers)

    def flat_deep(self, B: int, device) -> torch.Tensor:
        # Return the deepest GFC centroid flattened for classifier feature concatenation
        if len(self.maps) == 0:
            return None
        return self.maps[-1].detach().flatten(1).expand(B, -1).to(device)


# Global CNN autoencoder (CNN-ImageAE): continually adapts frozen patch features via depth-wise adapter layers
class GlobalDecoder(nn.Module):
    def __init__(self, channel: int, feat_map: int, n_parts=4, exp=5,
                 img_size=32, n_layer_adapter=1, use_gfc=False, num_classes=2):
        super().__init__()

        if isinstance(n_parts, (tuple, list)):
            self.rows, self.cols = int(n_parts[0]), int(n_parts[1])
        else:
            n_parts = int(n_parts)
            sq = math.isqrt(n_parts)
            if sq * sq != n_parts:
                raise ValueError(f"n_parts={n_parts} must be a perfect square.")
            self.rows, self.cols = sq, sq

        self.n_parts         = self.rows * self.cols
        self.n_parts_arg     = n_parts
        self.exp             = exp
        self.channel         = channel
        self.feat_map        = feat_map
        self.n_layer_adapter = n_layer_adapter
        self.use_gfc         = use_gfc

        if isinstance(img_size, int):
            self.img_h = self.img_w = img_size
        else:
            self.img_h, self.img_w = img_size

        F = feat_map

        self.adapter0 = nn.Sequential(
            nn.Conv2d(F,   F,   kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(F,   F,   kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(F,   F,   kernel_size=3, stride=1, padding=1), nn.ReLU(),
        )
        self.adapter1 = nn.Sequential(
            nn.Conv2d(F*2, F*2, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(F*2, F*2, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(F*2, F*2, kernel_size=3, stride=1, padding=1), nn.ReLU(),
        )
        self.adapter2 = nn.Sequential(
            nn.Conv2d(F*4, F*4, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(F*4, F*4, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(F*4, F*4, kernel_size=3, stride=1, padding=1), nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(F*4, F*2, kernel_size=3, stride=1), nn.ReLU(),
            nn.ConvTranspose2d(F*2, F,   kernel_size=3, stride=1), nn.ReLU(),
            nn.ConvTranspose2d(F,   channel, kernel_size=3, stride=1), nn.Sigmoid(),
        )

        if self.use_gfc:
            self.gfc = GFC()
        self._frozen_conv_cache = None

    def invalidate_frozen_cache(self):
        # Clear cached frozen conv weights to force recomputation
        self._frozen_conv_cache = None

    def cache_frozen_convs(self, sidenet_multi):
        # Build and cache padded copies of all frozen CNN-PatchAE conv weights
        frozen_convs = []
        for idx in range(self.n_parts):
            branch = getattr(sidenet_multi, f"cnn_p{self.n_parts_arg}_side{idx}")
            enc_convs = [m for m in branch.encoder.modules() if isinstance(m, nn.Conv2d)]
            branch_convs = []
            for orig_conv in enc_convs:
                padded = nn.Conv2d(
                    in_channels  = orig_conv.in_channels,
                    out_channels = orig_conv.out_channels,
                    kernel_size  = orig_conv.kernel_size,
                    stride       = orig_conv.stride,
                    padding      = 1,
                    bias         = orig_conv.bias is not None,
                ).to(orig_conv.weight.device)
                with torch.no_grad():
                    padded.weight.copy_(orig_conv.weight)
                    if orig_conv.bias is not None:
                        padded.bias.copy_(orig_conv.bias)
                for p in padded.parameters():
                    p.requires_grad_(False)
                branch_convs.append(padded)
            frozen_convs.append(branch_convs)
        self._frozen_conv_cache = frozen_convs

    def _get_adapted_layer_features(self, patch, frozen_branch_convs):
        # Apply frozen convs then adapter at each depth; return both adapter outputs and frozen outputs
        adapters      = [self.adapter0, self.adapter1, self.adapter2]
        repeat_counts = [1] + [self.n_layer_adapter] * (len(frozen_branch_convs) - 1)
        x             = patch
        adapted_feats = []
        frozen_feats  = []

        for frozen_conv, adapter, n_rep in zip(frozen_branch_convs, adapters, repeat_counts):
            x_frozen = F.relu(frozen_conv(x))
            frozen_feats.append(x_frozen.detach())
            x = x_frozen
            for _ in range(n_rep):
                x = adapter(x)
            adapted_feats.append(x)

        return adapted_feats, frozen_feats

    def _build_global_layer_features(self, x, sidenet_multi):
        # Extract per-patch features and tile them into full-image spatial maps at each depth
        B, C, H, W = x.shape
        ph = H // self.rows
        pw = W // self.cols

        if self._frozen_conv_cache is None:
            self.cache_frozen_convs(sidenet_multi)
        frozen_convs = self._frozen_conv_cache

        per_patch_adapted = []
        per_patch_frozen  = []
        for idx in range(self.n_parts):
            r, c  = divmod(idx, self.cols)
            patch = x[:, :, r*ph:(r+1)*ph, c*pw:(c+1)*pw]
            adapted, frozen = self._get_adapted_layer_features(patch, frozen_convs[idx])
            per_patch_adapted.append(adapted)
            per_patch_frozen.append(frozen)

        def tile(per_patch):
            n_depths = len(per_patch[0])
            tiled = []
            for d in range(n_depths):
                rows_concat = []
                for r in range(self.rows):
                    row = [per_patch[r * self.cols + c][d] for c in range(self.cols)]
                    rows_concat.append(torch.cat(row, dim=3))
                tiled.append(torch.cat(rows_concat, dim=2))
            return tiled

        global_layers = tile(per_patch_adapted)
        frozen_layers = tile(per_patch_frozen)

        return global_layers, frozen_layers

    def forward(self, x, sidenet_multi):
        # Forward pass: build frozen and adapter feature maps, decode bottleneck to reconstructed image
        global_layers, frozen_layers = self._build_global_layer_features(x, sidenet_multi)
        bottleneck     = global_layers[-1]
        reconstruction = self.decoder(bottleneck)

        if reconstruction.shape[-2:] != (self.img_h, self.img_w):
            reconstruction = F.interpolate(
                reconstruction, size=(self.img_h, self.img_w),
                mode='bilinear', align_corners=False)

        return reconstruction, bottleneck, global_layers, frozen_layers

    def get_classifier_features(self, x, sidenet_multi):
        # Extract flattened bottleneck concatenated with deepest GFC centroid for linear classifier
        with torch.no_grad():
            _, bottleneck, *_ = self.forward(x, sidenet_multi)
        B        = bottleneck.shape[0]
        bot_flat = bottleneck.flatten(1)
        if self.use_gfc:
            gfc_flat = self.gfc.flat_deep(B, bottleneck.device)
            return torch.cat([bot_flat, gfc_flat], dim=1)
        else:
            return bot_flat
