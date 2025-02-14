from chemprop.featurizers.bond import MultiHotBondFeaturizer
from rdkit.Chem.rdchem import Bond
from abc import ABC, abstractmethod
import numpy as np
from config.data_models import IFG
from typing import Optional, List


class BondFeatureExtender(ABC):
    def __init__(
        self,
        additional_feature_length: int,
        chemprop_featuriser: Optional[
            MultiHotBondFeaturizer
        ] = MultiHotBondFeaturizer(),
        chemprop_feature_length: Optional[int] = 14,
    ):
        self.chemprop_featuriser = chemprop_featuriser
        self.len = additional_feature_length + chemprop_feature_length

    def __len__(self):
        return self.len

    @abstractmethod
    def _additional_feature(self, bond: Bond) -> np.array:
        pass

    def __call__(self, a: Bond):
        additional_features = self._additional_feature(a)

        if self.chemprop_featuriser:
            base_feature = self.chemprop_featuriser(a)
            additional_features = np.concatenate([base_feature, additional_features])
            additional_features = np.array(additional_features, dtype=np.float32)

        return additional_features


class FGMembershipBondFeaturiser(BondFeatureExtender):
    def __init__(self, ifg_list: List[IFG], additional_feature_length: int = 3):
        super().__init__(
            additional_feature_length=additional_feature_length,
            chemprop_featuriser=MultiHotBondFeaturizer(),
            chemprop_feature_length=14,
        )
        self.ifg_list = ifg_list

    def _additional_feature(self, bond: Bond) -> np.array:
        bond_idx = bond.GetIdx()
        if any(bond_idx in ifg.bonds for ifg in self.ifg_list):
            return np.array([1, 0, 0])  # bond within a FG
        elif any(bond_idx in ifg.all_bonds for ifg in self.ifg_list):
            return np.array([0, 1, 0])  # bond between FG and non-FG
        else:
            return np.array([0, 0, 1])  # bond between non-FG
