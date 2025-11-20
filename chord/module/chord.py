import copy
import torch
from torch import nn
import torch.nn.functional as Fn
from torchvision.transforms import v2

from . import register, make
from .base import Base

from chord.util import fresnelSchlick, GeometrySchlickGGX, DistributionGGX
from chord.util import srgb_to_rgb, tone_gamma, get_positions, safe_01_div

class dummy_module(nn.Module):
    def forward(self, x): return x

def post_decoder(out_dict):
    out = {}
    for key in out_dict.keys():
        if key.startswith("approx"): continue
        elif key == "normal": 
            out[key] = Fn.normalize(2. * out_dict[key] - 1., dim=1) / 2. + 0.5
        elif key == "rou_met":
            out['roughness'], out['metalness'] = out_dict['rou_met'][:,0], out_dict['rou_met'][:,1]
        else: out[key] = out_dict[key]
    return out

def process_irradiance(radiance, kernel_size=25, res=64):
    """
    Process the irradiance using PyTorch, equivalent to the original OpenCV-based function.
    
    Args:
        radiance (torch.Tensor): Input radiance tensor (H, W).
        kernel_size (int): Size of the kernel for the median blur.
        res (int): Target resolution for resizing the image.
    
    Returns:
        torch.Tensor: Processed radiance tensor (res, res).
    """
    # Ensure the input radiance is a 4D tensor (B, 1, H, W)
    assert radiance.shape[1] == 1 and radiance.dim() == 4, f"Invalid radiance shape, got {radiance.shape}"
    # resize to low resolution
    resizer = v2.Resize(size=res, antialias=True)
    radiance = resizer(radiance)

    # Define a 11x11 averaging kernel
    kernel = torch.ones((1, 1, 11, 11), dtype=torch.float32).to(radiance) / 121.0
    # Apply convolution (averaging filter)
    radiance = Fn.pad(radiance, (5,)*4, mode="reflect")  # Pad for edge handling
    radiance = Fn.conv2d(radiance, kernel, padding=0)  # 'padding=2' to maintain input dimensions

    # Clamp values and scale to [0, 255] for median filtering
    radiance = torch.clamp(radiance * 255, 0, 255)  # Remove batch/channel dims

    # Apply median filtering
    paded_radiance = Fn.pad(radiance, (kernel_size // 2,) * 4, mode="reflect")  # Pad for edge handling
    unfolded = Fn.unfold(paded_radiance, kernel_size)  # Extract patches
    radiance = torch.median(unfolded, dim=1).values.view(radiance.shape)  # Median of patches

    # Normalize to [0, 1]
    rad_min, rad_max = radiance.amin([2,3], keepdim=True), radiance.amax([2,3], keepdim=True)
    radiance = (radiance - rad_min) / (rad_max - rad_min)
    return radiance

def opt_light_dir(_radiance, _num_samples=6):
    '''
        _radiance: (bs, 1, h, w)
    '''
    assert _radiance.shape[1] == 1 and _radiance.dim()==4
    bs, _, h, w = _radiance.shape

    def evenly_sample(_num_samples, min=0, max=2*torch.pi):
        # returns torch.tensor([1, _num_samples])
        return torch.tensor(range(_num_samples+1)) * (max - min) / _num_samples + min

    def compute_radiance_diff(angles):
        num = angles.shape[-1]
        dirs = torch.cat([torch.cos(angles), torch.sin(angles)]).T
        pos_dir = grid_pos.repeat(num, 1, 1, 1)
        pos_mask = torch.einsum("abcd,ad->abc", pos_dir, dirs) > 0
        neg_mask = torch.einsum("abcd,ad->abc", pos_dir, dirs) < 0
        samples_radiance = _radiance.repeat(1,num,1,1)
        radiance_diff = (samples_radiance*pos_mask[None] - samples_radiance*neg_mask[None]).sum([2,3])
        return radiance_diff
    
    angle_min, angle_max = 0, 2*torch.pi
    grid_pos = Fn.normalize(get_positions(h,w,10)[...,:2], dim=-1, eps=1e-6).to(_radiance)
    while(((angle_max - angle_min) > (torch.pi/90))):
        angles = evenly_sample(_num_samples, angle_min, angle_max)[None].to(_radiance)
        diffs = compute_radiance_diff(angles).mean(0)
        angle_min = angles[:,diffs.argmax()].item() - (angle_max - angle_min)/_num_samples
        angle_max = angles[:,diffs.argmax()].item() + (angle_max - angle_min)/_num_samples

    light_angle = angles[:, diffs.argmax()]
    return torch.tensor([torch.cos(light_angle), torch.sin(light_angle)]).to(_radiance)


def find_light_dir(raw_irradiance, light):
    raw_irradiance = v2.functional.rgb_to_grayscale(raw_irradiance)
    irradiance = process_irradiance(raw_irradiance)
    dir = opt_light_dir(irradiance)
    dir = torch.cat([dir, torch.tensor([0.5**0.5]).to(dir)])
    _light = copy.deepcopy(light)
    _light.direction = dir
    return _light

@register("chord")
class Chord(Base):
    def setup(self):
        # Define forward chain
        self.chain_type = self.config.get("chain_type", "chord")
        self.chain = self.config.get("chain_library", {})[self.chain_type]
        self.prompts = self.config.get("rgbx_prompts", {})
        self.roughness_step = self.config.get("roughness_step", 10)
        self.metallic_step = self.config.get("metallic_step", 0.2)

        self.sd = make(self.config.stable_diffusion.name, self.config.stable_diffusion)
        self.dtype = self.sd.dtype
        self.device = self.sd.device

        # LEGO-conditioning
        self.sd.unet.ConvIns = nn.ModuleDict()
        self.sd.unet.ConvOuts = nn.ModuleDict()
        self.sd.unet.FirstDownBlocks = nn.ModuleDict()
        self.sd.unet.LastUpBlocks = nn.ModuleDict()
        for key in list(set("_".join(self.chain.values()).split("_"))) + ["noise"]:
            if "0" in key or "1" in key: continue
            self.sd.unet.ConvIns[key] = nn.Conv2d(4, 320, 3, 1 , 1, device=self.device, dtype=self.dtype)
            self.sd.unet.ConvIns[key].load_state_dict(self.sd.unet.conv_in.state_dict())
        for kout in list(set(self.chain.keys())):
            self.sd.unet.ConvOuts[kout] = nn.Conv2d(320, 4, 3, 1 , 1, device=self.device, dtype=self.dtype)
            self.sd.unet.ConvOuts[kout].load_state_dict(self.sd.unet.conv_out.state_dict())
            self.sd.unet.LastUpBlocks[kout] = copy.deepcopy(self.sd.unet.up_blocks[-1]).to(self.device)
            self.sd.unet.FirstDownBlocks[kout] = copy.deepcopy(self.sd.unet.down_blocks[0]).to(self.device)
        self.sd.unet.ConvIns.train()
        self.sd.unet.ConvOuts.train()
        self.sd.unet.FirstDownBlocks.train()
        self.sd.unet.LastUpBlocks.train()
        self.sd.unet.conv_in = dummy_module()
        self.sd.unet.conv_out = dummy_module()

        # Load Lights
        if self.config.get("prior_light", None) is None:
            self.prior_light = make("point-light", {"position": [0, 0, 10]})
        else:
            self.prior_light = make(self.config.prior_light.name, self.config.prior_light)

        # Init Embeddings
        self.text_emb = {}
    # Eq.3
    def compute_approxIrr(self, render, basecolor):
        approxIrr = safe_01_div.apply(srgb_to_rgb(render), srgb_to_rgb(basecolor))
        return tone_gamma(approxIrr)
    # Eq.6
    @torch.no_grad()
    def compute_approxRouMet(self, render, maps, seperate=False, light=None):
        render = srgb_to_rgb(render)
        bs, _, h, w = render.shape
        light = find_light_dir(maps['approxIrr'], self.prior_light) if light is None else light
        # light.direction = estimate_light_dir(render, maps)
        pos = get_positions(h, w, 10).to(self.device)
        cameras = torch.tensor([0, 0, 10.0]).to(self.device)

        # sample grid
        r_samples = torch.arange(25, 225+self.roughness_step, self.roughness_step) / 255
        m_samples = torch.arange(0., 1.+self.metallic_step, self.metallic_step)

        grid_maps = {} # change map size into: gs, bs, h, w, c
        grid_maps['basecolor'] = maps['basecolor'][None].permute(0,1,3,4,2)
        grid_maps['normal'] = maps['normal'][None].permute(0,1,3,4,2)
        r_values = r_samples[:,None].repeat(1,len(m_samples)).reshape(-1,1,1,1,1).to(maps['basecolor'])
        m_values = m_samples[None].repeat(len(r_samples),1).reshape(-1,1,1,1,1).to(maps['basecolor'])
        # split into chunks to avoid OOM
        chunk_size = 25
        rgb_list, r_list, m_list = [], [], []
        for _r, _m in zip(torch.split(r_values, chunk_size), torch.split(m_values, chunk_size)):
            grid_maps['roughness'], grid_maps['metallic'] = _r, _m
            _rgb = self.compute_render(grid_maps, cameras, pos, light)
            loss = (render[None].permute(0,1,3,4,2) - _rgb).abs().sum(-1,keepdim=True)
            min_idx = loss.argmin(dim=0,keepdim=True)
            r_list.append(torch.gather(grid_maps['roughness'].flatten(), 0, min_idx.flatten()).reshape(min_idx.shape))
            m_list.append(torch.gather(grid_maps['metallic'].flatten(), 0, min_idx.flatten()).reshape(min_idx.shape))
            rgb_list.append(torch.gather(_rgb, 0, min_idx.repeat(1,1,1,1,3)))
        rgb = torch.cat(rgb_list).permute(0,1,4,2,3)
        roughness = torch.cat(r_list).permute(0,1,4,2,3)
        metallic = torch.cat(m_list).permute(0,1,4,2,3)
        loss = (render[None] - rgb).abs().sum(2,keepdim=True)
        roughness = torch.gather(roughness, 0, loss.argmin(dim=0,keepdim=True))[0]
        metallic = torch.gather(metallic, 0, loss.argmin(dim=0,keepdim=True))[0]
        torch.cuda.empty_cache()
        if seperate:
            return roughness, metallic
        else:
            out = torch.cat([roughness, metallic, torch.zeros_like(roughness)], dim=1)
            return out


    @torch.no_grad()
    def compute_render(self, maps, camera_position, pos, light):
        '''
            maps: gs, bs, h, w, c (gs: the number of grids)
        '''
        def cos(x, y): 
            return torch.clamp((x*y).sum(-1, keepdim=True), min=0, max=1)
        
        # pre-process
        albedo = srgb_to_rgb(maps['basecolor'])
        normal = maps['normal'].clone()
        normal[..., :2] = normal[..., [1,0]]
        N = Fn.normalize((normal - 0.5) * 2.0, dim=-1, eps=1e-6)
        roughness = maps['roughness']
        metallic = maps['metallic']
        V = Fn.normalize(camera_position - pos, dim=-1, eps=1e-6).repeat(1,1,1,1,1).to(self.device)
        irradiance, L = light(pos)
        irradiance, L = irradiance.repeat(1,1,1,1,1).to(self.device), L.repeat(1,1,1,1,1).to(self.device)
        # rendering
        H = Fn.normalize(L+V, dim=-1, eps=1e-6)
        f0 = torch.ones_like(albedo).to(self.device) * 0.04
        F0 = torch.lerp(f0, albedo, metallic)
        F = fresnelSchlick(cos(H,V), F0)   
        ks = F

        diffuse = (1-ks) * albedo / torch.pi
        diffuse *= 1-metallic

        NDF = DistributionGGX(cos(N,H), roughness) 
        G = GeometrySchlickGGX(cos(N,L), roughness) * GeometrySchlickGGX(cos(N,V), roughness)

        numerator = NDF * G * F
        denominator = 4.0 * cos(N,V) * cos(N,L) + 1e-3
        specular = numerator / denominator
        ambient = 0.3 * albedo

        rgb = (diffuse + specular) * irradiance * cos(N,L) + ambient

        return rgb
    
    def forward(self, maps:dict):
        # prepare
        bs = maps['render'].shape[0]
        self.sd.scheduler.set_timesteps(1)
        t = self.sd.scheduler.timesteps[0]
        # chain processing
        pred, pred_latent, arxiv_latent = {}, {}, {}
        for kout, info in self.chain.items():
            info = info.split("_")
            keys, ids = info[:-1], info[-1]
            # Swap active LEGO blocks
            self.sd.unet.down_blocks[0] = self.sd.unet.FirstDownBlocks[kout]
            self.sd.unet.up_blocks[-1] = self.sd.unet.LastUpBlocks[kout]
            # Eq.2, summing input latents
            in_latent = 0
            for k, i in zip(keys, ids):
                if i=="0":
                    if not k in arxiv_latent.keys(): arxiv_latent[k] = self.sd.encode_imgs_deterministic(maps[k])
                    zx = arxiv_latent[k]
                else:
                    zx = pred_latent[k]
                in_latent += self.sd.unet.ConvIns[k](zx)
            in_latent = in_latent / len(keys)
            # single-step denoising
            embs = self.produce_embeddings(kout, bs)
            out_latent = self.sd.unet(in_latent, t, **embs)[0]
            out_latent = self.sd.unet.ConvOuts[kout](out_latent)            
            pred_latent[kout] = self.sd.scheduler.step(out_latent, t, torch.zeros_like(zx)).pred_original_sample
            pred[kout] = self.sd.decode_latents(pred_latent[kout]).float()
            # compute intermediate representations
            if self.chain_type in ["chord"] and kout == "basecolor":
                pred['approxIrr'] = self.compute_approxIrr(maps['render'], pred['basecolor'])
                pred_latent['approxIrr'] = self.sd.encode_imgs_deterministic(pred['approxIrr'])
            if self.chain_type in ["chord"] and kout == "normal":
                pred['approxRM'] = self.compute_approxRouMet(maps['render'], pred, seperate=False)
                pred_latent['approxRM'] = self.sd.encode_imgs_deterministic(pred['approxRM'])

        return pred     
    
    @torch.no_grad()
    def produce_embeddings(self, key, batch_size):
        if key not in self.text_emb.keys():
            self.text_emb[key] = self.sd.encode_text(self.prompts[key], "max_length")
        prompt_emb = self.text_emb[key].expand(batch_size, -1, -1)
        return { "encoder_hidden_states": prompt_emb }