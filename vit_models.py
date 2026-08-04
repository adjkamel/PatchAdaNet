
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

#----------------------- Build the 2D positional grid for patch tokens.
def _make_grid(n_parts):
    if isinstance(n_parts, (list, tuple)):
        return int(n_parts[0]), int(n_parts[1])
    n  = int(n_parts)
    sq = math.isqrt(n)
    assert sq * sq == n, f"n_parts={n} must be a perfect square or (rows,cols) tuple"
    return sq, sq

#----------------------- Two-layer feed-forward block used inside a Transformer block.
class MLP(nn.Module):
    #----------------------- Initialize the MLP.
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    #----------------------- Forward pass.
    def forward(self, x):
        return self.net(x)

#----------------------- Multi-head self-attention module.
class Attention(nn.Module):
    #----------------------- Initialize the Attention.
    def __init__(self, dim, heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    #----------------------- Forward pass.
    def forward(self, x):
        xn = self.norm(x)
        out, _ = self.attn(xn, xn, xn)
        return x + out

#----------------------- Pre-norm Transformer encoder block (attention + MLP).
class TransformerBlock(nn.Module):
    #----------------------- Initialize the TransformerBlock.
    def __init__(self, dim, heads=4, mlp_ratio=4):
        super().__init__()
        self.attn = Attention(dim, heads)
        self.mlp  = MLP(dim, dim * mlp_ratio)
        self.norm = nn.LayerNorm(dim)

    #----------------------- Forward pass.
    def forward(self, x):
        x = self.attn(x)
        x = x + self.mlp(self.norm(x))
        return x

#----------------------- Per-depth learnable token sequences acting as GFC anchors across tasks.
class PrototypeMemory(nn.Module):

    #----------------------- Initialize the PrototypeMemory.
    def __init__(self, momentum: float = 0.9):
        super().__init__()
        self.momentum = momentum

        self.tokens = nn.ParameterList()
        self.register_buffer('is_initialised', torch.tensor(False))

    #----------------------- Custom loading to rebuild the ParameterList from a checkpoint.
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

        tok_keys = sorted(
            [k for k in state_dict.keys() if k.startswith(prefix + 'tokens.')],
            key=lambda x: int(x.split('.')[-1])
        )
        if tok_keys:
            self.tokens = nn.ParameterList()
            for key in tok_keys:
                param = nn.Parameter(state_dict[key].float())
                self.tokens.append(param)
            self.is_initialised.fill_(True)
            print(f"  [PrototypeMemory-ViT] Loaded {len(tok_keys)} "
                  f"prototype token sequences from state_dict.")

    @torch.no_grad()
    #----------------------- Task 1: warm-start prototypes from frozen enc_block tokens per depth.
    def initialise(self, frozen_seqs: list):
        self.tokens = nn.ParameterList()
        for fs in frozen_seqs:

            mean_tok = fs.detach().mean(dim=0, keepdim=True)
            self.tokens.append(nn.Parameter(mean_tok.clone()))
        self.is_initialised.fill_(True)
        print(f"  [PrototypeMemory-ViT] Initialised {len(frozen_seqs)} depth "
              f"prototypes from task-1 FROZEN enc_block outputs.")

    @torch.no_grad()
    #----------------------- Task t>1: EMA-update prototypes from last adapter_block outputs.
    def update(self, adapter_out_seqs: list):
        if len(self.tokens) == 0:
            raise RuntimeError(
                "PrototypeMemory not initialised. "
                "Call initialise() first or load from state_dict."
            )
        for proto_tok, ao in zip(self.tokens, adapter_out_seqs):
            new_mean = ao.detach().mean(dim=0, keepdim=True).float().to(proto_tok.device)
            proto_tok.data.mul_(self.momentum).add_(new_mean * (1.0 - self.momentum))
        print(f"  [PrototypeMemory-ViT] Updated {len(self.tokens)} depth prototypes "
              f"from ADAPTER outputs (momentum={self.momentum}).")

    #----------------------- Pull each adapter_block output toward its prototype during training.
    def alignment_loss(self, adapter_out_seqs: list) -> torch.Tensor:
        if len(self.tokens) == 0:
            return torch.tensor(0.0, device=adapter_out_seqs[0].device)

        loss = torch.tensor(0.0, device=adapter_out_seqs[0].device)
        for ao, proto_tok in zip(adapter_out_seqs, self.tokens):
            B  = ao.shape[0]
            pt = proto_tok.detach().expand(B, -1, -1)
            loss = loss + F.mse_loss(ao, pt.type_as(ao))
        return loss / len(adapter_out_seqs)

    #----------------------- Return the deepest prototype token sequence flattened for the classifier.
    def flat_deep(self, B: int, device) -> torch.Tensor:
        if len(self.tokens) == 0:
            return None
        return self.tokens[-1].detach().flatten(1).expand(B, -1).to(device)

#----------------------- Single-patch ViT autoencoder branch.
class SideViT(nn.Module):
    #----------------------- Initialize the SideViT.
    def __init__(self, channel, patch_h, patch_w,
                 patch_size=4, dim=128, n_blocks=2, heads=4):
        super().__init__()
        self.channel    = channel
        self.patch_h    = patch_h
        self.patch_w    = patch_w
        self.patch_size = patch_size
        self.dim        = dim
        self.n_blocks   = n_blocks

        self.n_tokens_h = patch_h // patch_size
        self.n_tokens_w = patch_w // patch_size
        self.n_tokens   = self.n_tokens_h * self.n_tokens_w
        token_in_dim    = channel * patch_size * patch_size

        self.patch_embed = nn.Linear(token_in_dim, dim)
        self.pos_embed   = nn.Parameter(torch.randn(1, self.n_tokens, dim) * 0.02)
        self.enc_blocks  = nn.ModuleList(
            [TransformerBlock(dim, heads) for _ in range(n_blocks)]
        )
        self.enc_norm    = nn.LayerNorm(dim)
        self.dec_blocks  = nn.ModuleList(
            [TransformerBlock(dim, heads) for _ in range(n_blocks)]
        )
        self.dec_norm    = nn.LayerNorm(dim)
        self.dec_proj    = nn.Linear(dim, token_in_dim)

    #----------------------- Split an image into flattened patch tokens.
    def patchify(self, x):
        B, C, H, W = x.shape
        ps = self.patch_size
        nh, nw = H // ps, W // ps
        x = x.reshape(B, C, nh, ps, nw, ps)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, nh * nw, C * ps * ps)
        return x

    #----------------------- Reassemble patch tokens back into an image.
    def unpatchify(self, tokens):
        B, N, D = tokens.shape
        C, ps   = self.channel, self.patch_size
        nh, nw  = self.n_tokens_h, self.n_tokens_w
        x = tokens.reshape(B, nh, nw, C, ps, ps)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(B, C, nh * ps, nw * ps)
        return x

    #----------------------- Run the encoder blocks and return per-depth token sequences.
    def encode_blocks(self, x):
        tokens = self.patch_embed(self.patchify(x)) + self.pos_embed
        block_outputs = []
        for blk in self.enc_blocks:
            tokens = blk(tokens)
            block_outputs.append(tokens)
        return block_outputs

    #----------------------- Forward pass.
    def forward(self, x):
        block_outputs = self.encode_blocks(x)
        enc_tokens    = self.enc_norm(block_outputs[-1])
        dec = enc_tokens
        for blk in self.dec_blocks:
            dec = blk(dec)
        dec   = self.dec_norm(dec)
        recon = self.unpatchify(self.dec_proj(dec))
        return recon, enc_tokens

#----------------------- Container of per-position SideViT branches over the patch grid.
class SideViTMulti(nn.Module):
    #----------------------- Initialize the SideViTMulti.
    def __init__(self, channel, img_h, img_w,
                 n_parts=16, patch_size=4, dim=128,
                 n_blocks=2, heads=4, exp=6):
        super().__init__()
        self.rows, self.cols = _make_grid(n_parts)
        self.n_parts         = self.rows * self.cols
        self.channel         = channel
        self.exp             = exp
        self.n_parts_arg     = n_parts
        self.patch_h         = img_h // self.rows
        self.patch_w         = img_w // self.cols

        for idx in range(self.n_parts):
            setattr(self, f"vit_p{n_parts}_side{idx}",
                    SideViT(channel, self.patch_h, self.patch_w,
                            patch_size, dim, n_blocks, heads))

    #----------------------- Split an input image into the non-overlapping patch grid.
    def _extract_patches(self, x):
        patches = []
        for r in range(self.rows):
            for c in range(self.cols):
                patches.append(
                    x[:, :, r*self.patch_h:(r+1)*self.patch_h,
                             c*self.patch_w:(c+1)*self.patch_w])
        return patches

    #----------------------- Forward pass.
    def forward(self, x):
        patches = self._extract_patches(x)
        recons, enc_tokens_list = [], []
        for idx in range(self.n_parts):
            side  = getattr(self, f"vit_p{self.n_parts_arg}_side{idx}")
            recon, enc_tokens = side(patches[idx])
            recons.append(recon)
            enc_tokens_list.append(enc_tokens)
        return recons, enc_tokens_list

#----------------------- Full-image ViT autoencoder with adapters and cross-task PrototypeMemory.
class GlobalViTAE(nn.Module):

    #----------------------- Initialize the GlobalViTAE.
    def __init__(self, channel, img_h, img_w,
                 n_parts=16, patch_size=4, dim=128,
                 n_blocks=2, heads=4, exp=6, n_layer_adapter=1,use_proto=False, supervised=False, num_classes=2):
        super().__init__()
        self.rows, self.cols = _make_grid(n_parts)
        self.n_parts     = self.rows * self.cols
        self.channel     = channel
        self.img_h       = img_h
        self.img_w       = img_w
        self.patch_size  = patch_size
        self.dim         = dim
        self.n_blocks    = n_blocks
        self.exp         = exp
        self.n_layer_adapter = n_layer_adapter
        self.n_parts_arg = n_parts
        self.use_proto=use_proto
        self.supervised=supervised

        cell_h = img_h // self.rows
        cell_w = img_w // self.cols
        self.n_cell_tokens = (cell_h // patch_size) * (cell_w // patch_size)
        self.N_total       = self.n_parts * self.n_cell_tokens

        self.frozen_pos_embeds = nn.ParameterList([
            nn.Parameter(torch.randn(1, self.N_total, dim) * 0.02)
            for _ in range(n_blocks)
        ])

        self.enc_blocks = nn.ModuleList(
            [TransformerBlock(dim, heads) for _ in range(n_blocks)]
        )
        self.adapter_blocks = nn.ModuleList(
            [TransformerBlock(dim, heads) for _ in range(n_blocks * n_layer_adapter)]
        )
        self.enc_norm = nn.LayerNorm(dim)

        token_in_dim     = channel * patch_size * patch_size
        cell_h_tokens    = cell_h // patch_size
        cell_w_tokens    = cell_w // patch_size
        self.dec_n_tok_h = self.rows * cell_h_tokens
        self.dec_n_tok_w = self.cols * cell_w_tokens
        assert self.dec_n_tok_h * self.dec_n_tok_w == self.N_total

        self.dec_blocks = nn.ModuleList(
            [TransformerBlock(dim, heads) for _ in range(n_blocks)]
        )
        self.dec_norm = nn.LayerNorm(dim)
        self.dec_proj = nn.Linear(dim, token_in_dim)

        if self.use_proto:
            self.prototype = PrototypeMemory()
        self._frozen_token_cache = None

        if self.supervised and num_classes is not None:
            self.down_features=nn.Linear(128*256, 1024)
            self.classifier=nn.Linear(1024, num_classes)

    #----------------------- Clear the cached frozen token activations.
    def invalidate_frozen_cache(self):
        self._frozen_token_cache = None

    #----------------------- Compute the frozen per-depth token sequences.
    def _build_frozen_tokens(self, x, sidenet_multi):
        ph = self.img_h // self.rows
        pw = self.img_w // self.cols

        per_cell = []
        for idx in range(self.n_parts):
            r, c  = divmod(idx, self.cols)
            patch = x[:, :, r*ph:(r+1)*ph, c*pw:(c+1)*pw]
            side  = getattr(sidenet_multi, f"vit_p{self.n_parts_arg}_side{idx}")
            with torch.no_grad():
                block_outputs = side.encode_blocks(patch)
            per_cell.append([bo.detach() for bo in block_outputs])

        frozen_seqs = []
        for d in range(self.n_blocks):
            tok_seq = torch.cat([per_cell[k][d] for k in range(self.n_parts)], dim=1)
            tok_seq = tok_seq + self.frozen_pos_embeds[d]
            frozen_seqs.append(tok_seq)

        return frozen_seqs

    #----------------------- Reassemble patch tokens back into an image.
    def unpatchify(self, tokens):
        B, N, D = tokens.shape
        C, ps   = self.channel, self.patch_size
        nh, nw  = self.dec_n_tok_h, self.dec_n_tok_w
        x = tokens.reshape(B, nh, nw, C, ps, ps)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(B, C, nh * ps, nw * ps)
        return x

    #----------------------- Forward pass.
    def forward(self, x, sidenet_multi):
        frozen_seqs = self._build_frozen_tokens(x, sidenet_multi)

        adapter_out_seqs = []

        out = self.enc_blocks[0](frozen_seqs[0])

        for k in range(0, self.n_layer_adapter):
            out = self.adapter_blocks[k](out)

        adapter_out_seqs.append(out)

        for d in range(1, self.n_blocks):
            out = self.enc_blocks[d](frozen_seqs[d] + out)

            for k in range((d * self.n_layer_adapter), (d * self.n_layer_adapter+ self.n_layer_adapter)):
                out = self.adapter_blocks[k](out)

            adapter_out_seqs.append(out)

        last_adp        = self.adapter_blocks[-1]
        bottleneck      = last_adp.mlp(last_adp.norm(out))
        enc_out         = self.enc_norm(out)

        dec = enc_out
        for blk in self.dec_blocks:
            dec = blk(dec)
        dec   = self.dec_norm(dec)
        dec   = self.dec_proj(dec)
        recon = self.unpatchify(dec)

        if recon.shape[-2:] != (self.img_h, self.img_w):
            recon = F.interpolate(recon, size=(self.img_h, self.img_w),
                                  mode='bilinear', align_corners=False)

        B               = out.shape[0]
        bottleneck_flat = bottleneck.reshape(B, -1)

        if self.supervised and hasattr(self, 'classifier'):
            bottleneck_flat_m=self.down_features(bottleneck_flat)
            out_classifier=self.classifier(bottleneck_flat_m)
            return recon, bottleneck_flat, adapter_out_seqs, frozen_seqs, out_classifier

        return recon, bottleneck_flat, adapter_out_seqs, frozen_seqs

    #----------------------- Concatenate bottleneck and deepest prototype for the classifier.
    def get_classifier_features(self, x, sidenet_multi):
        with torch.no_grad():

             _, bottleneck_flat, *_= self.forward(x, sidenet_multi)

        B          = bottleneck_flat.shape[0]
        if self.use_proto:
            proto_flat = self.prototype.flat_deep(B, bottleneck_flat.device)
            return torch.cat([bottleneck_flat, proto_flat], dim=1)
        else:
            return bottleneck_flat
