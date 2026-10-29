"""Minimal parameter-shared on-policy MAPPO (central V, local stochastic actor).

Scope: train one team against frozen scripted opponents. It is a real PPO update,
not a claim of competitively strong or multi-population self-play.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal, Categorical

LOG_TWO = 0.6931471805599453


class SharedActor(nn.Module):
    def __init__(self, obs_dim, hidden=96):
        super().__init__()
        self.body=nn.Sequential(nn.Linear(obs_dim,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh())
        self.mean=nn.Linear(hidden,2)
        self.log_std=nn.Parameter(torch.full((2,),-.5))
        self.split=nn.Linear(hidden,2)
        self.eject=nn.Linear(hidden,2)

    def distribution(self, obs, split_mask, eject_mask):
        h=self.body(obs)
        mean=self.mean(h)
        std=self.log_std.clamp(-4,1).exp().expand_as(mean)
        split_logits=self.split(h).masked_fill(~split_mask.bool(), -1e9)
        eject_logits=self.eject(h).masked_fill(~eject_mask.bool(), -1e9)
        return Normal(mean,std),Categorical(logits=split_logits),Categorical(logits=eject_logits)

    def act(self, obs, split_mask, eject_mask, deterministic=False):
        gauss,split,eject=self.distribution(obs,split_mask,eject_mask)
        raw=gauss.mean if deterministic else gauss.rsample()
        action_move=torch.tanh(raw)
        action_split=split.probs.argmax(-1) if deterministic else split.sample()
        action_eject=eject.probs.argmax(-1) if deterministic else eject.sample()
        logp=self.logprob(obs,split_mask,eject_mask,action_move,action_split,action_eject)
        return action_move,action_split,action_eject,logp

    def logprob(self,obs,split_mask,eject_mask,move,split_action,eject_action):
        gauss,split,eject=self.distribution(obs,split_mask,eject_mask)
        clipped=move.clamp(-1+1e-6,1-1e-6)
        raw=torch.atanh(clipped)
        log_det=2*(LOG_TWO-raw-torch.nn.functional.softplus(-2*raw))
        continuous=(gauss.log_prob(raw)-log_det).sum(-1)
        return continuous+split.log_prob(split_action.long())+eject.log_prob(eject_action.long())


class TeamCritic(nn.Module):
    def __init__(self, state_dim, hidden=96):
        super().__init__()
        self.model=nn.Sequential(nn.Linear(state_dim,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,1))
    def forward(self,state):
        return self.model(state).squeeze(-1)


class MAPPO:
    def __init__(self,obs_dim,state_dim,lr=3e-4,gamma=.99,lam=.95,clip=.2,epochs=3,device='cpu'):
        self.device=device
        self.actor=SharedActor(obs_dim).to(device)
        self.critic=TeamCritic(state_dim).to(device)
        self.optimizer=torch.optim.Adam(list(self.actor.parameters())+list(self.critic.parameters()),lr=lr)
        self.gamma,self.lam,self.clip,self.epochs=gamma,lam,clip,epochs

    @torch.no_grad()
    def choose(self,obs,deterministic=False):
        features=torch.tensor(obs['features'],dtype=torch.float32,device=self.device).unsqueeze(0)
        s_mask=torch.tensor(obs['split_mask'].astype(bool),device=self.device).unsqueeze(0)
        e_mask=torch.tensor(obs['eject_mask'].astype(bool),device=self.device).unsqueeze(0)
        m,s,e,lp=self.actor.act(features,s_mask,e_mask,deterministic=deterministic)
        action={'move':m[0].cpu().numpy().astype(np.float32),
                'split':int(s.item()),'eject':int(e.item())}
        return action,float(lp.item())

    def update(self,records):
        if not records:
            raise ValueError('no trajectory')
        batches=[]
        for trajectory in records:
            next_adv=0.
            for index in range(len(trajectory)-1,-1,-1):
                item=trajectory[index]
                nonterminal=0. if item['done'] else 1.
                next_v=item['next_value']
                delta=item['reward'] + self.gamma*nonterminal*next_v-item['value']
                next_adv=delta+self.gamma*self.lam*nonterminal*next_adv
                item['advantage']=next_adv
                item['target']=next_adv+item['value']
            batches.extend(trajectory)
        features=torch.as_tensor(np.stack([x['obs'] for x in batches]),dtype=torch.float32)
        states=torch.as_tensor(np.stack([x['state'] for x in batches]),dtype=torch.float32)
        sm=torch.as_tensor(np.stack([x['split_mask'] for x in batches]),dtype=torch.bool)
        em=torch.as_tensor(np.stack([x['eject_mask'] for x in batches]),dtype=torch.bool)
        moves=torch.as_tensor(np.stack([x['move'] for x in batches]),dtype=torch.float32)
        splits=torch.as_tensor([x['split'] for x in batches],dtype=torch.long)
        ejects=torch.as_tensor([x['eject'] for x in batches],dtype=torch.long)
        old_lp=torch.as_tensor([x['logp'] for x in batches],dtype=torch.float32)
        advantages=torch.as_tensor([x['advantage'] for x in batches],dtype=torch.float32)
        returns=torch.as_tensor([x['target'] for x in batches],dtype=torch.float32)
        advantages=(advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-6)
        metrics={}
        for _ in range(self.epochs):
            new_lp=self.actor.logprob(features,sm,em,moves,splits,ejects)
            ratio=torch.exp((new_lp-old_lp).clamp(-30,30))
            surrogate1=ratio*advantages
            surrogate2=ratio.clamp(1-self.clip,1+self.clip)*advantages
            actor_loss=-torch.minimum(surrogate1,surrogate2).mean()
            predictions=self.critic(states)
            value_loss=((predictions-returns)**2).mean()
            loss=actor_loss+.5*value_loss
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite PPO loss')
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.actor.parameters())+list(self.critic.parameters()),.5)
            self.optimizer.step()
            metrics={'loss':float(loss.item()),'actor_loss':float(actor_loss.item()),
                     'value_loss':float(value_loss.item()), 'approx_kl':float((old_lp-new_lp).mean().item()),
                     'batch_size':len(batches)}
        return metrics

    def save(self,path,metadata=None):
        """Atomic versioned checkpoint with checksum sidecar."""
        from pathlib import Path
        import os
        import tempfile
        from training.artifacts import sha256_file, atomic_write_bytes
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
        os.close(fd)
        try:
            torch.save({'actor':self.actor.state_dict(),'critic':self.critic.state_dict(),
                    'optimizer':self.optimizer.state_dict(),
                    'obs_dim':self.actor.body[0].in_features,
                    'state_dim':self.critic.model[0].in_features,
                    'hyperparameters':{'gamma':self.gamma,'lam':self.lam,'clip':self.clip,
                                       'epochs':self.epochs},
                    'checkpoint_schema': 2,
                    'metadata':metadata or {}}, tmp_path)
            os.replace(tmp_path, path)
            atomic_write_bytes(str(path)+'.sha256', (sha256_file(path)+'\n').encode('ascii'))
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @classmethod
    def load(cls,path,*,allow_legacy=False):
        from training.artifacts import verify_checksum
        verify_checksum(path,allow_legacy=allow_legacy)
        obj=torch.load(path,map_location='cpu',weights_only=True)
        version=obj.get('checkpoint_schema',1)
        if version not in (1,2):
            raise ValueError(f'unsupported checkpoint schema {version}')
        if version == 1 and not allow_legacy:
            raise ValueError('legacy checkpoint requires allow_legacy=True')
        hp=obj.get('hyperparameters',{})
        model=cls(obj['obs_dim'],obj['state_dim'],gamma=hp.get('gamma',.99),
                  lam=hp.get('lam',.95),clip=hp.get('clip',.2),epochs=hp.get('epochs',3))
        model.actor.load_state_dict(obj['actor'])
        model.critic.load_state_dict(obj['critic'])
        model.optimizer.load_state_dict(obj['optimizer'])
        return model,obj.get('metadata',{})
