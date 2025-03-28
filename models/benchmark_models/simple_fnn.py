import torch.nn as nn
import torch


class SimpleFNN(nn.Module):
    def __init__(self, input_dim: int, dropout=0.2, hidden_dim=128, output_dim=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, batch):
        features = batch["polymer_feats"]
        fingerprints = batch["fingerprints_tensor"]
        # x_avg = fingerprints.view(32, 2048, 2).mean(dim=2)
        # print(f"features: {features.shape}, fingerprint: {fingerprints.shape}")

        x = torch.cat([features, fingerprints], dim=1)
        x = x.view(x.shape[0], -1)  # Flatten for FNN
        return self.model(x)

    def combine_fingerprint_features(
        self, rdkit: torch.Tensor, polymer_mapping: torch.Tensor
    ):
        """
        Combines RDKit features into a [B, 2*D] tensor by averaging monomer features
        and appending solvent features per polymer system.

        Args:
            rdkit (torch.Tensor): [N, D] RDKit features for all molecules
            polymer_mapping (torch.Tensor): [N] mapping from molecule to polymer system

        Returns:
            torch.Tensor: [B, 2*D] combined features (monomer_avg || solvent)
        """
        device = rdkit.device
        num_polymers = polymer_mapping.max().item() + 1
        feature_dim = rdkit.shape[1]

        # Tags: monomer = 0, solvent = 1 (solvent is last in each polymer system)
        solvent_tags = torch.zeros_like(polymer_mapping, dtype=torch.bool)
        sorted_mapping, _ = torch.sort(polymer_mapping)
        _, _, counts = torch.unique_consecutive(sorted_mapping, return_counts=True)
        solvent_indices = torch.cumsum(counts, dim=0) - 1
        solvent_tags[solvent_indices] = True

        monomer_mask = ~solvent_tags
        solvent_mask = solvent_tags

        # Prepare output tensor
        combined = torch.zeros((num_polymers, 2 * feature_dim), device=device)

        for i in range(num_polymers):
            indices = polymer_mapping == i

            monomer_feats = rdkit[indices & monomer_mask].sum(dim=0, keepdim=True)
            solvent_feat = rdkit[indices & solvent_mask].sum(dim=0, keepdim=True)
            combined[i] = (monomer_feats + solvent_feat).squeeze(0)

            # monomer_avg = monomer_feats.mean(dim=0)

            # solvent_avg = solvent_feat.mean(dim=0)
            # combined[i] = torch.cat([monomer_avg, solvent_feat.squeeze(0)], dim=-1)

        return combined


class SimpleFNNRDKit(nn.Module):
    def __init__(self, input_dim: int, dropout=0.2, hidden_dim=128, output_dim=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, batch):
        features = batch["polymer_feats"]
        rdkit = batch["rdkit_tensor"]
        polymer_mapping = batch["polymer_mapping"]
        pooled_rdkit = self.combine_rdkit_features(rdkit, polymer_mapping)
        x = torch.cat([features, pooled_rdkit], dim=1)
        x = x.view(x.shape[0], -1)  # Flatten for FNN
        return self.model(x)

    def combine_rdkit_features(
        self, rdkit: torch.Tensor, polymer_mapping: torch.Tensor
    ):
        """
        Combines RDKit features into a [B, 2*D] tensor by averaging monomer features
        and appending solvent features per polymer system.

        Args:
            rdkit (torch.Tensor): [N, D] RDKit features for all molecules
            polymer_mapping (torch.Tensor): [N] mapping from molecule to polymer system

        Returns:
            torch.Tensor: [B, 2*D] combined features (monomer_avg || solvent)
        """
        device = rdkit.device
        num_polymers = polymer_mapping.max().item() + 1
        feature_dim = rdkit.shape[1]

        # Tags: monomer = 0, solvent = 1 (solvent is last in each polymer system)
        solvent_tags = torch.zeros_like(polymer_mapping, dtype=torch.bool)
        _, _, counts = torch.unique_consecutive(polymer_mapping, return_counts=True)
        solvent_indices = torch.cumsum(counts, dim=0) - 1
        solvent_tags[solvent_indices] = True

        monomer_mask = ~solvent_tags
        solvent_mask = solvent_tags

        # Prepare output tensor
        combined = torch.zeros((num_polymers, 2 * feature_dim), device=device)

        for i in range(num_polymers):
            indices = polymer_mapping == i

            monomer_feats = rdkit[indices & monomer_mask]
            solvent_feat = rdkit[indices & solvent_mask]

            monomer_avg = monomer_feats.mean(dim=0)
            combined[i] = torch.cat([monomer_avg, solvent_feat.squeeze(0)], dim=-1)

        return combined
