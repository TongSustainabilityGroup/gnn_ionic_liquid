import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdmolops

import torch
from torch import nn
from torcheval.metrics import R2Score
import torch_geometric
from torch_geometric.nn import global_mean_pool, GATv2Conv
import pytorch_lightning as pl

import warnings
warnings.filterwarnings("ignore")

pl.seed_everything(0)
torch.manual_seed(0)

#=====data processing=====

sort_by = 'cation_smiles'
sheet_name = 'Glycolysis_Final'
df0 = pd.read_excel('PET_IL_Data.xlsx', sheet_name=sheet_name) 
df0 = df0.sort_values(by=[sort_by], ascending=True)[df0.columns[:22]]
df_orig = df0.copy()
df0 = df0.drop(['catalyst1', 'solvent'], axis=1)

df = df0.copy()
df['catalyst_amount'].fillna(0, inplace=True)
df['PET_source'].fillna('NaN', inplace=True) 
df['PET_size_mm'].fillna('NaN', inplace=True) 
df['cation_name'].fillna('NotUsed', inplace=True)
df['anion_name'].fillna('NotUsed', inplace=True) 
df_idx = df['yield'].isna()
df.loc[df_idx, 'yield'] = df[df_idx]['conversion'] * df[df_idx]['selectivity'] / 100
df['PET_source'].fillna('NaN', inplace=True) 
df['catalyst_amount'] = df['catalyst_amount'] / df['PET_amount']
df['solvent_amount'] = df['solvent_amount'] / df['PET_amount']
df['PET_size_mm'] = df['PET_size_mm'].astype(float)

df = df.loc[df.loc[:,'yield'] > 0 ]
df['target'] = df['yield']
df.dropna(subset=['target'], inplace=True)
df2 = df.copy()
df = df.drop(['cation_smiles','anion_smiles', 'conversion', 'selectivity', 'yield', 'PET_amount', 'IL_smiles',],axis=1)

ani_price = {
    'ZnCl3': 1200,
    'CoCl4': 6800,
    'CrCl4': 10000,
    'CuCl3': 8200,
    'CuCl4': 8200,
    'Ala': 3900,
    'Ser': 3400,
    'Ac': 386,
    'Gly': 2000,
    'For': 535,
    'Asp': 6800,
    'MnCl3': 1400,
    'Co(Ac)3': 9000,
    'CoCl3': 6900,
    'Cl': 0,
    'Br':0,
    'Lys': 1270,
    'FeCl4': 2000,
    'ZnCl4': 1200,
    'Zn(Ac)3': 2100,
    'OH': 0,
    'Fe2Cl6O': 2000,
    'Pro': 18000,
    'Cu(Ac)3': 5110,
    'PO4': 870,
    'His': 44000,
    'Leu': 8700,
    'NiCl4': 6700,
    'Arg': 9400,
    'Mn(Ac)3': 2600,
    'Ni(Ac)3': 9670,
    'Try': 9500,
    'But': 1321,
    'HCO3': 250,
    'HSO4': 250,
    'Im': 8343,
    'Mesy': 1600
    }

cat_price = {
    'AMIM': 2760,
    'C6TMG': 8000, 
    'Ch': 3000,
    'C2TMG': 6000, 
    'TMG': 5000, 
    'N2222': 8290,
    'N1111': 4000,
    'C4TMG': 7000, 
    'C8TMG': 9000, 
    'BMIM': 2000,
    'HMIM': 2760,
    'DMIM': 2600,
    'DEIM': 2600,
    'EMIM': 2300,
    'UREA': 300,
}

cation_list = np.unique(df2[['cation_name','cation_smiles']].values.astype(str),axis=0)
cation_list = np.char.replace(cation_list, ' ', '')
cation_dict = dict(zip(cation_list[:,0], cation_list[:,1]))
anion_list =  np.unique(df2[['anion_name','anion_smiles']].values.astype(str),axis=0)
anion_list =  np.char.replace(anion_list, ' ', '')
anion_dict = dict(zip(anion_list[:,0], anion_list[:,1]))
mol_dict = {}
for i in cation_dict.keys():
    i_mol = Chem.MolFromSmiles(cation_dict[i])
    mol_dict[i] = (i, cation_dict[i], rdmolops.GetFormalCharge(i_mol), Descriptors.MolWt(i_mol), cat_price[i], 1)
for i in anion_dict.keys():
    i_mol = Chem.MolFromSmiles(anion_dict[i])
    mol_dict[i] = (i, anion_dict[i], rdmolops.GetFormalCharge(i_mol), Descriptors.MolWt(i_mol), ani_price[i], 0)
cation_dict_inv = {v: k for k, v in cation_dict.items()}
anion_dict_inv = {v: k for k, v in anion_dict.items()}


#=====model=====

# source: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/smiles.html
x_map = {
    'atomic_num':
    list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
        'CHI_TETRAHEDRAL',
        'CHI_ALLENE',
        'CHI_SQUAREPLANAR',
        'CHI_TRIGONALBIPYRAMIDAL',
        'CHI_OCTAHEDRAL',
    ],
    'degree':
    list(range(0, 11)),
    'formal_charge':
    list(range(-5, 7)),
    'num_hs':
    list(range(0, 9)),
    'num_radical_electrons':
    list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
    'mw': [  1.01,   4.  ,   6.94,   9.01,  10.81,  12.01,  14.01,  16.  ,
        19.  ,  20.18,  22.99,  24.31,  26.98,  28.09,  30.97,  32.06,
        35.45,  39.95,  39.1 ,  40.08,  44.96,  47.87,  50.94,  52.  ,
        54.94,  55.85,  58.93,  58.69,  63.55,  65.38,  69.72,  72.63,
        74.92,  78.97,  79.9 ,  83.8 ,  85.47,  87.62,  88.91,  91.22,
        92.91,  95.95,  97.  , 101.07, 102.91, 106.42, 107.87, 112.41,
       114.82, 118.71, 121.76, 127.6 , 126.9 , 131.29, 132.91, 137.33,
       138.91, 140.12, 140.91, 144.24, 145.  , 150.36, 151.96, 157.25,
       158.93, 162.5 , 164.93, 167.26, 168.93, 173.05, 174.97, 178.49,
       180.95, 183.84, 186.21, 190.23, 192.22, 195.08, 196.97, 200.59,
       204.38, 207.21, 208.98, 209.  , 210.  ],
}

e_map = {
    'bond_type': [
        'UNSPECIFIED',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'QUADRUPLE',
        'QUINTUPLE',
        'HEXTUPLE',
        'ONEANDAHALF',
        'TWOANDAHALF',
        'THREEANDAHALF',
        'FOURANDAHALF',
        'FIVEANDAHALF',
        'AROMATIC',
        'IONIC',
        'HYDROGEN',
        'THREECENTER',
        'DATIVEONE',
        'DATIVE',
        'DATIVEL',
        'DATIVER',
        'OTHER',
        'ZERO',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOANY',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
    ],
    'is_conjugated': [False, True],
}


# source: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/smiles.html
def from_smiles(smiles: str, with_hydrogen: bool = False,
                kekulize: bool = False) -> 'torch_geometric.data.Data':
    from rdkit import Chem, RDLogger

    from torch_geometric.data import Data

    RDLogger.DisableLog('rdApp.*')

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        mol = Chem.MolFromSmiles('')
    if with_hydrogen:
        mol = Chem.AddHs(mol)
    if kekulize:
        Chem.Kekulize(mol)

    xs = []
    for atom in mol.GetAtoms():
        x = []
        x.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
        x.append(x_map['formal_charge'].index(atom.GetFormalCharge()))
        x.append(x_map['hybridization'].index(str(atom.GetHybridization())))
        x.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))
        x.append(x_map['mw'][atom.GetAtomicNum()-1])
        xs.append(x)

    x = torch.tensor(xs, dtype=torch.long).view(-1, 5)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        e = []
        e.append(e_map['bond_type'].index(str(bond.GetBondType())))
        e.append(e_map['stereo'].index(str(bond.GetStereo())))
        e.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))

        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 3)

    if edge_index.numel() > 0:  # Sort indices.
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)


class ILDataModule(pl.LightningDataModule):
    def __init__(self, df, batch_size):
        super().__init__()
   
        self.df = df.copy().fillna(0)
        self.df = self.df.replace(['Bottle', 'Pellet', 'Powder', 'NaN'], [-1.,0.,1.,0.])
        self.batch_size = batch_size
        self.df0 = self.df.copy()

        self.stat_dict = {
            'inp_mean': self.df[['catalyst_amount', 'solvent_amount', 'temperature_c', 'reaction_time_min','PET_source','PET_size_mm']].mean(),
            'inp_std': self.df[['catalyst_amount', 'solvent_amount', 'temperature_c', 'reaction_time_min','PET_source','PET_size_mm']].std(),
            'out_mean': self.df['target'].mean(), 
            'out_std': self.df['target'].std()
        }
        
        self.df[['catalyst_amount', 'solvent_amount', 'temperature_c', 'reaction_time_min','PET_source','PET_size_mm']] = (
            (self.df[['catalyst_amount', 'solvent_amount', 'temperature_c', 'reaction_time_min','PET_source','PET_size_mm']]
            - self.stat_dict['inp_mean']) / self.stat_dict['inp_std']
        )
        
        self.df['target'] = (self.df['target'] - self.stat_dict['out_mean']) / self.stat_dict['out_std']
        

        IL_list = np.unique(self.df['IL_smiles'])
        self.IL_dict = {i: from_smiles(i) for i in IL_list}      

    def prep(self, x):
        data_point = self.IL_dict[x['IL_smiles']].clone()
        data_point.x = data_point.x.to(torch.float)
        data_point.cx = torch.tensor(x[['catalyst_amount', 'solvent_amount', 
                        'temperature_c', 'reaction_time_min','PET_source','PET_size_mm']].to_list()).to(torch.float).expand(1,-1)
        data_point.y = torch.tensor(x['target']).to(torch.float)
        return data_point

    def setup(self, stage=None):
        if (stage == 'fit' or stage is None):
            self.df['data'] = self.df.apply(self.prep, axis=1)
            self.data_list = self.df['data'].tolist()

    def train_dataloader(self):
        return torch_geometric.loader.DataLoader(self.data_list, batch_size=self.batch_size, shuffle=True, drop_last=True) 


class GNN_Model(torch.nn.Module):
    def __init__(self):
        super(GNN_Model, self).__init__()
        self.conv1 = GATv2Conv(5, 16, 4, edge_dim=3) 
        self.conv2 = GATv2Conv(64, 32, 4, edge_dim=3) 
        self.conv3 = GATv2Conv(128, 32, 4, edge_dim=3, dropout=.4)
        self.conv4 = GATv2Conv(128, 32, 2, edge_dim=3)
        self.lin = nn.Linear(64, 32)

    def forward(self, x, edge_index, edge_features, batch):
        x = self.conv1(x, edge_index, edge_features)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_features)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_features)
        x = x.relu()
        x = self.conv4(x, edge_index, edge_features)
        x = x.relu()
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x


class FCN_Model(torch.nn.Module):
    def __init__(self, n_tasks):
        super(FCN_Model, self).__init__()
        self.n_tasks = n_tasks
        
        hidden_dim = [n_tasks+6, 128, 128, 64, 32]
        activation = nn.Sigmoid
        self.layer = nn.Sequential()
        for i in range(len(hidden_dim)-1):
            self.layer.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))
            self.layer.append(activation())
        self.layer.append(nn.Linear(hidden_dim[-1], 1))

    def forward(self, x):
        x = self.layer(x)
        return x
    

class ILPredictor(pl.LightningModule):
    def __init__(self):
        super().__init__()
      
        self.gnn_model = GNN_Model()
        self.fcn_model = FCN_Model(n_tasks=32)
        self.loss = nn.MSELoss()
        self.lossl1 = nn.L1Loss()
        self.metric = R2Score(multioutput="variance_weighted")
        self.test_list =[]
        self.target_list =[]

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.parameters(), lr=8e-4,weight_decay=1e-4)
        self.lr_scheduler = self.configure_schduler(optimizer)
        return optimizer

    def configure_schduler(self, optimizer): 
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10000, T_mult=1, eta_min=3e-5, last_epoch=-1, verbose=False) 
        return scheduler

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()  

    def forward(self, x, edge_index, cx, edge_features, batch_index):
        gnn_out = self.gnn_model(x, edge_index, edge_features, batch_index) 
        fcn_in = torch.hstack([gnn_out, cx]) 
        fcn_out = self.fcn_model(fcn_in) 
        return fcn_out.squeeze()

    def training_step(self, batch, batch_idx):
        x, cx, edge_index, edge_features = batch.x, batch.cx, batch.edge_index, batch.edge_attr
        batch_index = batch.batch
        x_out = self(x, edge_index, cx, edge_features, batch_index)

        loss = self.loss(x_out, batch.y)

        pred = (x_out.cpu() * ILdata.stat_dict['out_std']) + ILdata.stat_dict['out_mean']
        target_test = (batch.y.cpu() * ILdata.stat_dict['out_std']) + ILdata.stat_dict['out_mean']
        slossl1 = self.lossl1(pred, target_test)
        slossl2 = self.loss(pred, target_test)**0.5

        self.metric.reset()

        self.metric.update(pred, target_test)
        R2 = self.metric.compute()

        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/lossl1', slossl1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/lossl2', slossl2, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/R2", R2, on_step=False, on_epoch=True, prog_bar=True)
        return loss


#=====train=====
logger = pl.loggers.TensorBoardLogger(
        save_dir="./", 
        name="tsboard_logs",
        version="train",
    )

gmodel = ILPredictor()
torch.set_float32_matmul_precision('medium')
trainer = pl.Trainer(
    max_steps=120_000,
    gradient_clip_val=5,
    check_val_every_n_epoch=10,
    logger=logger,
)
ILdata = ILDataModule(df2, batch_size=8)
ILdata.setup()

trainer.fit(gmodel, ILdata.train_dataloader())
trainer.save_checkpoint("train.ckpt")
