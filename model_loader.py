import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv
import xgboost as xgb
import joblib

# The GNN architecture MUST be defined in the script that loads the weights
class StabilizedSupplyChainGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, edge_types):
        super().__init__()
        self.conv1 = HeteroConv({edge: SAGEConv((-1, -1), hidden_channels) for edge in edge_types}, aggr='mean')
        self.conv2 = HeteroConv({edge: SAGEConv((-1, -1), hidden_channels) for edge in edge_types}, aggr='mean')
        self.lin = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
        return self.lin(x_dict['order'])

def load_full_pipeline():
    # 1. Load Mappings
    assets = joblib.load('app_assets.pkl')
    ship_map = assets['ship_map']
    clean_feature_cols = assets['clean_feature_cols']

    # 2. Initialize and Load GNN
    # Note: edge_types must match your notebook exactly
    edge_types = [('order', 'uses', 'ship_mode'), ('ship_mode', 'rev_uses', 'order')]
    gnn_model = StabilizedSupplyChainGNN(hidden_channels=128, out_channels=2, edge_types=edge_types)
    gnn_model.load_state_dict(torch.load('supply_chain_gnn.pt', map_location=torch.device('cpu')))
    gnn_model.eval()

    # 3. Load XGBoost
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model('hybrid_xgb_model.json')

    return gnn_model, xgb_model, ship_map, clean_feature_cols