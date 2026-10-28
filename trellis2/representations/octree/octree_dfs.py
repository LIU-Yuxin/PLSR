import torch
import torch.nn as nn
import torch.nn.functional as F


class DfsOctree:
    """
    Sparse Voxel Octree (SVO) implementation for PyTorch.
    Using Depth-First Search (DFS) order to store the octree.
    DFS order suits rendering and ray tracing.

    The structure and data are separatedly stored.
    Structure is stored as a continuous array, each element is a 3*32 bits descriptor.
    |-----------------------------------------|
    |      0:3 bits      |      4:31 bits     |
    |      leaf num      |       unused       |
    |-----------------------------------------|
    |               0:31  bits                |
    |                child ptr                |
    |-----------------------------------------|
    |               0:31  bits                |
    |                data ptr                 |
    |-----------------------------------------|
    Each element represents a non-leaf node in the octree.
    The valid mask is used to indicate whether the children are valid.
    The leaf mask is used to indicate whether the children are leaf nodes.
    The child ptr is used to point to the first non-leaf child. Non-leaf children descriptors are stored continuously from the child ptr.
    The data ptr is used to point to the data of leaf children. Leaf children data are stored continuously from the data ptr.

    There are also auxiliary arrays to store the additional structural information to facilitate parallel processing.
      - Position: the position of the octree nodes.
      - Depth: the depth of the octree nodes.

    Args:
        depth (int): the depth of the octree.
    """

    def __init__(
            self,
            depth,
            aabb=[0,0,0,1,1,1],
            sh_degree=2,
            primitive='voxel',
            primitive_config={},
            device='cuda',
        ):
        self.max_depth = depth
        self.aabb = torch.tensor(aabb, dtype=torch.float32, device=device)
        self.device = device
        self.sh_degree = sh_degree
        self.active_sh_degree = sh_degree
        self.primitive = primitive
        self.primitive_config = primitive_config

        self.structure = torch.tensor([[8, 1, 0]], dtype=torch.int32, device=self.device)
        self.position = torch.zeros((8, 3), dtype=torch.float32, device=self.device)
        self.depth = torch.zeros((8, 1), dtype=torch.uint8, device=self.device)
        self.position[:, 0] = torch.tensor([0.25, 0.75, 0.25, 0.75, 0.25, 0.75, 0.25, 0.75], device=self.device)
        self.position[:, 1] = torch.tensor([0.25, 0.25, 0.75, 0.75, 0.25, 0.25, 0.75, 0.75], device=self.device)
        self.position[:, 2] = torch.tensor([0.25, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75], device=self.device)
        self.depth[:, 0] = 1

        self.data = ['position', 'depth']
        self.param_names = []

        if primitive == 'voxel':
            self.features_dc = torch.zeros((8, 1, 3), dtype=torch.float32, device=self.device)
            self.features_ac = torch.zeros((8, (sh_degree+1)**2-1, 3), dtype=torch.float32, device=self.device)
            self.data += ['features_dc', 'features_ac']
            self.param_names += ['features_dc', 'features_ac']
            if not primitive_config.get('solid', False):
                self.density = torch.zeros((8, 1), dtype=torch.float32, device=self.device)
                self.data.append('density')
                self.param_names.append('density')
        elif primitive == 'gaussian':
            self.features_dc = torch.zeros((8, 1, 3), dtype=torch.float32, device=self.device)
            self.features_ac = torch.zeros((8, (sh_degree+1)**2-1, 3), dtype=torch.float32, device=self.device)
            self.opacity = torch.zeros((8, 1), dtype=torch.float32, device=self.device)
            self.data += ['features_dc', 'features_ac', 'opacity']
            self.param_names += ['features_dc', 'features_ac', 'opacity']
        elif primitive == 'trivec':
            self.trivec = torch.zeros((8, primitive_config['rank'], 3, primitive_config['dim']), dtype=torch.float32, device=self.device)
            self.density = torch.zeros((8, primitive_config['rank']), dtype=torch.float32, device=self.device)
            self.features_dc = torch.zeros((8, primitive_config['rank'], 1, 3), dtype=torch.float32, device=self.device)
            self.features_ac = torch.zeros((8, primitive_config['rank'], (sh_degree+1)**2-1, 3), dtype=torch.float32, device=self.device)
            self.density_shift = 0
            self.data += ['trivec', 'density', 'features_dc', 'features_ac']
            self.param_names += ['trivec', 'density', 'features_dc', 'features_ac']
        elif primitive == 'decoupoly':
            self.decoupoly_V = torch.zeros((8, primitive_config['rank'], 3), dtype=torch.float32, device=self.device)
            self.decoupoly_g = torch.zeros((8, primitive_config['rank'], primitive_config['degree']), dtype=torch.float32, device=self.device)
            self.density = torch.zeros((8, primitive_config['rank']), dtype=torch.float32, device=self.device)
            self.features_dc = torch.zeros((8, primitive_config['rank'], 1, 3), dtype=torch.float32, device=self.device)
            self.features_ac = torch.zeros((8, primitive_config['rank'], (sh_degree+1)**2-1, 3), dtype=torch.float32, device=self.device)
            self.density_shift = 0
            self.data += ['decoupoly_V', 'decoupoly_g', 'density', 'features_dc', 'features_ac']
            self.param_names += ['decoupoly_V', 'decoupoly_g', 'density', 'features_dc', 'features_ac']

        self.setup_functions()

    def setup_functions(self):
        self.density_activation = (lambda x: torch.exp(x - 2)) if self.primitive != 'trivec' else (lambda x: x)
        self.opacity_activation = lambda x: torch.sigmoid(x - 6)
        self.inverse_opacity_activation = lambda x: torch.log(x / (1 - x)) + 6
        self.color_activation = lambda x: torch.sigmoid(x)

    
    @property
    def num_leaf_nodes(self):
        return self.depth.shape[0]

    @property
    def cur_depth(self):
        return self.depth.max().item()
    
    @property
    def occupancy(self):
        return self.num_leaf_nodes / 8 ** self.cur_depth
    
    @property
    def get_xyz(self):
        return self.position

    @property
    def get_depth(self):
        return self.depth

    @property
    def get_density(self):
        if self.primitive == 'voxel' and self.primitive_config.get('solid', False):
            return torch.full((self.position.shape[0], 1), torch.finfo(torch.float32).max, dtype=torch.float32, device=self.device)
        return self.density_activation(self.density)
    
    # UPSTREAM_REVIEW_NOTE (Microsoft TRELLIS):
    # Automated code review flagged a possible inherited bug: Gaussian
    # opacity reads density although Gaussian primitives store opacity.
    # Upstream behavior is preserved; see trellis/representations/octree/octree_dfs.py.
    @property
    def get_opacity(self):
        return self.opacity_activation(self.density)

    @property
    def get_trivec(self):
        return self.trivec

    @property
    def get_decoupoly(self):
        return F.normalize(self.decoupoly_V, dim=-1), self.decoupoly_g


    @property
    def get_features(self):
        if self.sh_degree == 0:
            return self.features_dc
        return torch.cat([self.features_dc, self.features_ac], dim=-2)

    def state_dict(self):
        ret = {'structure': self.structure, 'position': self.position, 'depth': self.depth, 'sh_degree': self.sh_degree, 'active_sh_degree': self.active_sh_degree, 'primitive_config': self.primitive_config, 'primitive': self.primitive}
        if hasattr(self, 'density_shift'):
            ret['density_shift'] = self.density_shift
        for data in set(self.data + self.param_names):
            if not isinstance(getattr(self, data), nn.Module):
                ret[data] = getattr(self, data)
            else:
                ret[data] = getattr(self, data).state_dict()
        return ret

    def load_state_dict(self, state_dict):
        keys = list(set(self.data + self.param_names + list(state_dict.keys()) + ['structure', 'position', 'depth']))
        for key in keys:
            if key not in state_dict:
                print(f"Warning: key {key} not found in the state_dict.")
                continue
            try:
                if not isinstance(getattr(self, key), nn.Module):
                    setattr(self, key, state_dict[key])
                else:
                    getattr(self, key).load_state_dict(state_dict[key])
            except Exception as e:
                print(e)
                raise ValueError(f"Error loading key {key}.")
