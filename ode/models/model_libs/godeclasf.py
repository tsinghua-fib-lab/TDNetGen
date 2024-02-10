from ode.models.model_libs.odeblock import *
from ode.utils import *
from ode.models.model_libs.classifier import *
from ode.models.model_libs.base_modules import *

class GodeClassifier(nn.Module):
    
    def __init__(self, config):

        super().__init__()
        self.config = config
        self.traj_encoder = TrajEncoder(self.config)
        self.traj_decoder = TrajDecoder(self.config)
        self.odeblock = ODEBlock(self.config)
        self.classifier = ResilienceClassifier(self.config)
        
        self.classification_steps = self.config.classification_steps
    
    def transfer_to_adjacent(self, X: th.Tensor, E: th.Tensor):

       
        A = E[:, :, :, 1:].sum(dim=-1).float()
        return A
        
    
    def forward(self, x_lo, x_hi, X_in, E_in, t_in, t_eval, extra_data, node_mask):
        
        node_features = extra_data.X
        node_features = node_features.unsqueeze(2)
        node_features = node_features.repeat(1, 1, self.time_ticks, 1)
        
        if self.config.general.use_encoder:
            x = self.encoder(x_lo)
            self.ode.ode_func.update_graph(X_in, E_in)
            x = th.cat([x, node_features], dim=-1)
            # print(x.shape)
            # assert False
            x = self.ode(t_eval, x)
            x_lo_out = self.decoder(x)
        else:
            self.ode.ode_func.update_graph(X_in, E_in)
            x = self.ode(t_eval, x_lo)
            x_lo_out = x.flatten(start_dim=2)
        
        if self.config.general.use_encoder:
            x = self.encoder(x_hi)
            self.ode.ode_func.update_graph(X_in, E_in)
            x = th.cat([x, node_features], dim=-1)
            # print(x.shape)
            # assert False
            x = self.ode(t_eval, x)
            x_hi_out = self.decoder(x)
        else:
            self.ode.ode_func.update_graph(X_in, E_in)
            x = self.ode(t_eval, x_hi)
            x_hi_out = x.flatten(start_dim=2)
        
        x_lo_clasf_input = x_lo_out[:, :, -self.classification_steps:]
        x_hi_clasf_input = x_hi_out[:, :, -self.classification_steps:]
        
        if self.config.is_trm:
            clasf_feat = th.cat([x_lo_clasf_input, x_hi_clasf_input], dim=1) # (B, 2N, T)
        else:
            clasf_feat = th.cat([x_lo_clasf_input, x_hi_clasf_input], dim=2) # (B, N, 2T)
        
        adj = self.transfer_to_adjacent(X_in, E_in)

        y_hat = self.classifier(clasf_feat, adj, node_mask, t_in)
    
        return x_lo_out, x_hi_out, y_hat
    
    