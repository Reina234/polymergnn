from abc import ABC, abstractmethod
from chemprop.data import MolGraph
from tools.ertl_algorithm import ErtlAlgorithm
from dataloaders.featurisers import FGMembershipBondFeaturiser
from chemprop.featurizers.atom import MultiHotAtomFeaturizer
from chemprop.featurizers.molgraph.molecule import SimpleMoleculeMolGraphFeaturizer


class Mol2MolGraph(ABC):

    @abstractmethod
    def convert(self, mol) -> MolGraph:
        pass


class FGMembershipMol2MolGraph(Mol2MolGraph):

    def __init__(self, fg_detector=ErtlAlgorithm()):
        self.fg_detector = fg_detector

    def _retrieve_ifg_list(self, mol):
        return self.fg_detector.detect(mol)

    def convert(self, mol) -> MolGraph:
        atom_featuriser = MultiHotAtomFeaturizer.v2()
        bond_featuriser = FGMembershipBondFeaturiser(
            ifg_list=self._retrieve_ifg_list(mol)
        )
        featuriser = SimpleMoleculeMolGraphFeaturizer(
            atom_featurizer=atom_featuriser,
            bond_featurizer=bond_featuriser,
        )
        return featuriser(mol)
