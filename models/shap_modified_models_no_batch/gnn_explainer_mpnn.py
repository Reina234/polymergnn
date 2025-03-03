from chemprop.features.featurization import MolGraph, BatchMolGraph


def create_batch_molgraph(V, E):
    """
    Converts node (`V`) and edge (`E`) properties into a BatchMolGraph.

    Args:
        V: Node features (atoms).
        E: Edge index (bonds).

    Returns:
        BatchMolGraph object for Chemprop MPNN.
    """
    # Create a single `MolGraph` for the given molecule
    mol_graph = MolGraph(
        None
    )  # Chemprop requires a molecule, but we will inject edges manually

    # Add nodes (atoms)
    mol_graph.f_atoms = V.numpy().tolist()  # Convert to list format

    # Add edges (bonds)
    mol_graph.f_bonds = []  # Store bond features (Chemprop requires this)
    mol_graph.a2b = [[] for _ in range(len(V))]  # Adjacency list

    for i, j in zip(E[0].tolist(), E[1].tolist()):  # Iterate over edges
        mol_graph.a2b[i].append(len(mol_graph.f_bonds))  # Append bond index
        mol_graph.a2b[j].append(len(mol_graph.f_bonds))  # Undirected bond
        mol_graph.f_bonds.append([0] * 10)  # Placeholder for bond features

    # Wrap inside a `BatchMolGraph`
    batch_mol_graph = BatchMolGraph([mol_graph])

    return batch_mol_graph


class MoleculeEmbeddingModel(nn.Module):

    def __init__(
        self,
        chemprop_mpnn: ConfiguredMPNN,  # ConfiguredMPNN instance
        rdkit_featurizer: RDKitFeaturizer,
        selected_rdkit_features: List[str],
        chemberta_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,  # mpnn output dim
        use_rdkit: bool = True,
        use_chembert: bool = True,
    ):
        super().__init__()
        if not output_dim:
            output_dim = hidden_dim
        self.mpnn = chemprop_mpnn
        self.rdkit_featurizer = rdkit_featurizer if use_rdkit else None
        self.selected_rdkit_features = selected_rdkit_features if use_rdkit else None
        self.use_rdkit = use_rdkit
        self.use_chembert = use_chembert

        # If not using ChemBERTa, we set its dimension to zero.
        self.chemberta_dim = chemberta_dim if use_chembert else 0

        # RDKit dimension based on number of selected features.
        self.rdkit_dim = len(selected_rdkit_features) if use_rdkit else 0

        # The total input dimension is now:
        # MP-embedding + (ChemBERTa if used) + (RDKit features if used)
        total_in = self.mpnn.output_dim + 0

        self.bert_norm = nn.LayerNorm(chemberta_dim) if use_chembert else None
        self.rdkit_norm = (
            nn.LayerNorm(self.rdkit_dim) if (use_rdkit and self.rdkit_dim > 0) else None
        )

        self.fusion = nn.Sequential(
            nn.Linear(total_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.hidden_dim = hidden_dim

    def forward(self, V, E):
        molgraph = create_batch_molgraph(V, E)
        mpnn_out = self.mpnn(molgraph)

        self.fusion = nn.Sequential(
            nn.Linear(total_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        return molecule_embs, mpnn_out, chemberta_emb, rdkit_emb
