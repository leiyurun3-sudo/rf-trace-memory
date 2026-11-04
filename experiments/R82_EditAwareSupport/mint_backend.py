"""Existing offline FLUX backend: BF16 network, FP32 RF integration, VAE mode.

R54 uses rf_kv.invert and engine.execute. They both use actual scheduler
sigma; inversion checkpoints coincide exactly with generation nodes.
"""
import time
import numpy as np
import torch

class FluxBackend:
    def __init__(self,pipe,width=384,height=384,steps=28,max_length=128):
        self.pipe=pipe;self.width=width;self.height=height;self.steps=steps
        self.max_length=max_length;self.device=torch.device('cuda');self.dtype=torch.bfloat16
        self.nfe=0;self.decodes=0;self.encodes=0;self.last_trace=[]

    @torch.inference_mode()
    def encode(self,image):
        pixels=self.pipe.image_processor.preprocess(image,height=self.height,width=self.width).to(self.device,dtype=self.dtype)
        z=self.pipe.vae.encode(pixels).latent_dist.mode()
        z=(z-self.pipe.vae.config.shift_factor)*self.pipe.vae.config.scaling_factor
        h,w=z.shape[-2:]
        result=self.pipe._pack_latents(z,1,z.shape[1],h,w).float().cpu()
        ids=self.pipe._prepare_latent_image_ids(1,h//2,w//2,self.device,self.dtype).cpu()
        self.token_shape=(h//2,w//2);self.encodes+=1
        self.pipe.maybe_free_model_hooks()
        return result,ids

    @torch.inference_mode()
    def embeddings(self,prompt):
        self.pipe.maybe_free_model_hooks()
        torch.cuda.empty_cache()
        self.pipe._guidance_scale=3.5;self.pipe._joint_attention_kwargs=None
        em,po,ids=self.pipe.encode_prompt(prompt=prompt,prompt_2=prompt,device=self.device,num_images_per_prompt=1,max_sequence_length=self.max_length)
        result=tuple(x.detach().cpu() for x in (em,po,ids))
        self.pipe.maybe_free_model_hooks()
        return result

    def schedule(self,seq_len):
        from diffusers import FlowMatchEulerDiscreteScheduler
        from diffusers.pipelines.flux.pipeline_flux import calculate_shift
        scheduler=FlowMatchEulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
        cfg=scheduler.config
        mu=calculate_shift(seq_len,cfg.get('base_image_seq_len',256),cfg.get('max_image_seq_len',4096),cfg.get('base_shift',.5),cfg.get('max_shift',1.15))
        scheduler.set_timesteps(sigmas=np.linspace(1,1/self.steps,self.steps),mu=mu,device='cpu')
        sigmas=scheduler.sigmas.double().cpu()
        if len(sigmas)!=self.steps+1 or not bool(torch.all(sigmas[:-1]>sigmas[1:])) or abs(float(sigmas[-1]))>1e-8:
            raise ValueError('Unsupported sigma schedule')
        strength=getattr(self,'experiment_strength',1.)
        active=min(int(self.steps*strength),self.steps)
        if active<1:raise ValueError('strength leaves no active steps')
        return sigmas[self.steps-active:]

    @torch.inference_mode()
    def field(self,z,sigma,emb,ids,guidance):
        em,po,txt=tuple(x.to(self.device,dtype=self.dtype) for x in emb)
        g=torch.full((z.shape[0],),guidance,device=self.device,dtype=torch.float32) if self.pipe.transformer.config.guidance_embeds else None
        output=self.pipe.transformer(hidden_states=z.to(self.device,dtype=self.dtype),timestep=torch.full((z.shape[0],),float(sigma),device=self.device,dtype=self.dtype),
            guidance=g,pooled_projections=po,encoder_hidden_states=em,txt_ids=txt,img_ids=ids.to(self.device,dtype=self.dtype),return_dict=False)[0]
        self.nfe+=int(z.shape[0])
        return output.float()

    @torch.inference_mode()
    def decode(self,z):
        latent=self.pipe._unpack_latents(z.to(self.device,dtype=self.dtype),self.height,self.width,self.pipe.vae_scale_factor)
        latent=latent/self.pipe.vae.config.scaling_factor+self.pipe.vae.config.shift_factor
        image=self.pipe.vae.decode(latent,return_dict=False)[0]
        result=self.pipe.image_processor.postprocess(image,output_type='pil')[0];self.decodes+=1
        self.pipe.maybe_free_model_hooks()
        return result
