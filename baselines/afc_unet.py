"""AFC-Net."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=16):
        super().__init__()
        hidden=max(channels//ratio, 4)
        self.avg=nn.AdaptiveAvgPool2d(1); self.max=nn.AdaptiveMaxPool2d(1)
        self.mlp=nn.Sequential(nn.Conv2d(channels,hidden,1,bias=False),nn.ReLU(inplace=True),nn.Conv2d(hidden,channels,1,bias=False))
    def forward(self,x): return torch.sigmoid(self.mlp(self.avg(x))+self.mlp(self.max(x)))

class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super().__init__(); self.conv=nn.Conv2d(2,1,kernel_size,padding=kernel_size//2,bias=False)
    def forward(self,x):
        z=torch.cat([x.mean(1,keepdim=True),x.amax(1,keepdim=True)],1)
        return torch.sigmoid(self.conv(z))

class MFAG(nn.Module):
    """Gating attention plus channel/spatial refinement of encoder skip features."""
    def __init__(self,F_g,F_l,F_int):
        super().__init__()
        self.W_g=nn.Sequential(nn.Conv2d(F_g,F_int,1,bias=False),nn.BatchNorm2d(F_int))
        self.W_x=nn.Sequential(nn.Conv2d(F_l,F_int,1,bias=False),nn.BatchNorm2d(F_int))
        self.psi=nn.Sequential(nn.Conv2d(F_int,1,1),nn.Sigmoid())
        self.ca=ChannelAttention(F_l); self.sa=SpatialAttention()
    def forward(self,g,x):
        g1=self.W_g(g); x1=self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]: g1=F.interpolate(g1,size=x1.shape[2:],mode='bilinear',align_corners=False)
        gated=x*self.psi(F.relu(g1+x1,inplace=False))
        gated=gated*self.ca(gated); gated=gated*self.sa(gated)
        return gated

class CHATBlock(nn.Module):
    """Convolutional local path plus global self-attention and feed-forward block."""
    def __init__(self,dim,num_heads=4,mlp_ratio=4):
        super().__init__()
        if dim % num_heads: raise ValueError('dim must be divisible by num_heads')
        self.dim=dim; self.num_heads=num_heads; self.head_dim=dim//num_heads; self.scale=self.head_dim**-0.5
        self.norm1=nn.LayerNorm(dim); self.qkv=nn.Linear(dim,3*dim); self.proj=nn.Linear(dim,dim)
        self.local_conv=nn.Conv2d(dim,dim,3,padding=1,groups=dim)
        self.norm2=nn.LayerNorm(dim); self.mlp=nn.Sequential(nn.Linear(dim,dim*mlp_ratio),nn.GELU(),nn.Linear(dim*mlp_ratio,dim))
    def forward(self,x):
        b,c,h,w=x.shape
        tokens=x.flatten(2).transpose(1,2); qkv=self.qkv(self.norm1(tokens)).reshape(b,h*w,3,self.num_heads,self.head_dim).permute(2,0,3,1,4)
        q,k,v=qkv.unbind(0)
        attn=((q*self.scale)@k.transpose(-2,-1)).softmax(-1)
        glob=(attn@v).transpose(1,2).reshape(b,h*w,c)
        glob=self.proj(glob).transpose(1,2).reshape(b,c,h,w)
        x=x+glob+self.local_conv(x)
        t=x.flatten(2).transpose(1,2); t=t+self.mlp(self.norm2(t))
        return t.transpose(1,2).reshape(b,c,h,w)

class DoubleConv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__(); self.net=nn.Sequential(nn.Conv2d(in_ch,out_ch,3,padding=1,bias=False),nn.BatchNorm2d(out_ch),nn.ReLU(inplace=True),nn.Conv2d(out_ch,out_ch,3,padding=1,bias=False),nn.BatchNorm2d(out_ch),nn.ReLU(inplace=True))
    def forward(self,x): return self.net(x)

class FullScaleFusion(nn.Module):
    """Projects every encoder level to a shared channel count and spatial grid."""
    def __init__(self,encoder_channels,target_channels):
        super().__init__(); self.projections=nn.ModuleList([nn.Conv2d(c,target_channels,1) for c in encoder_channels]); self.fuse=nn.Sequential(nn.Conv2d(target_channels*len(encoder_channels),target_channels,3,padding=1,bias=False),nn.BatchNorm2d(target_channels),nn.ReLU(inplace=True))
    def forward(self,features,target_size):
        aligned=[]
        for feat,proj in zip(features,self.projections):
            z=proj(feat)
            if z.shape[2:]!=target_size: z=F.interpolate(z,size=target_size,mode='bilinear',align_corners=False)
            aligned.append(z)
        return self.fuse(torch.cat(aligned,dim=1))

class AFC_UNet(nn.Module):
    def __init__(self,in_channels=1,num_classes=2,base_c=32,num_heads=4):
        super().__init__()
        c=base_c
        self.inc=DoubleConv(in_channels,c)
        self.down1=nn.Sequential(nn.MaxPool2d(2),DoubleConv(c,2*c))
        self.down2=nn.Sequential(nn.MaxPool2d(2),DoubleConv(2*c,4*c))
        self.down3=nn.Sequential(nn.MaxPool2d(2),DoubleConv(4*c,8*c))
        self.bottleneck=nn.Sequential(nn.MaxPool2d(2),DoubleConv(8*c,16*c),CHATBlock(16*c,num_heads=num_heads))
        # Gate channels match the actual upsampled gating feature and skip feature.
        self.up3=nn.ConvTranspose2d(16*c,8*c,2,2); self.mfag3=MFAG(8*c,8*c,4*c); self.dec3=DoubleConv(16*c,8*c)
        self.up2=nn.ConvTranspose2d(8*c,4*c,2,2); self.mfag2=MFAG(4*c,4*c,2*c); self.dec2=DoubleConv(8*c,4*c)
        self.up1=nn.ConvTranspose2d(4*c,2*c,2,2); self.mfag1=MFAG(2*c,2*c,c); self.dec1=DoubleConv(4*c,2*c)
        self.up0=nn.ConvTranspose2d(2*c,c,2,2); self.dec0=DoubleConv(2*c,c)
        # Integrate full-scale fusion at the 1/4-resolution decoder level.
        self.full_scale=FullScaleFusion([c,2*c,4*c,8*c],4*c)
        self.fusion_refine=DoubleConv(8*c,4*c)
        self.outc=nn.Conv2d(c,num_classes,1)
    def forward(self,x):
        orig=x.shape[2:]
        x1=self.inc(x); x2=self.down1(x1); x3=self.down2(x2); x4=self.down3(x3); b=self.bottleneck(x4)
        g3=self.up3(b); d3=self.dec3(torch.cat([g3,self.mfag3(g3,x4)],1))
        g2=self.up2(d3); d2=self.dec2(torch.cat([g2,self.mfag2(g2,x3)],1))
        # Fuse all encoder scales on the x3 grid, then merge with decoder features.
        fs=self.full_scale([x1,x2,x3,x4],d2.shape[2:])
        d2=self.fusion_refine(torch.cat([d2,fs],1))
        g1=self.up1(d2); d1=self.dec1(torch.cat([g1,self.mfag1(g1,x2)],1))
        d0=self.up0(d1); d0=self.dec0(torch.cat([d0,x1],1))
        logits=self.outc(d0)
        if logits.shape[2:] != orig: logits=F.interpolate(logits,size=orig,mode='bilinear',align_corners=False)
        return logits

def build_model(in_channels=1,num_classes=2,base_c=32,**kwargs):
    return AFC_UNet(in_channels=in_channels,num_classes=num_classes,base_c=base_c,**kwargs)
