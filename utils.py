import numpy as np

from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.rdMolDescriptors import CalcTPSA
import pandas as pd
import warnings
from rdkit.Chem import rdmolops

warnings.filterwarnings(action='ignore')
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors


def read_csv(path, mode):
    df = pd.read_csv(path)  # ,encoding='gbk'

    smi_list = df['SMILES'].values
    Sol_list = df['label'].values

    if mode == 'train':
        split = df['label'].values
    else:
        split = df['CID'].values

    smi_ret = []
    sol_ret = []
    split_set = []
    imp_val_bool = True
    for (smi, sol, spl) in zip(smi_list, Sol_list, split):

        smi = smi.strip()
        iMol = Chem.MolFromSmiles(smi)
        # if "H" in smi:
        #     print("orin:",smi,"\n\n\n\n\n\n")
        #     iMol=Chem.RemoveHs(iMol)
        #     smi=Chem.MolToSmiles(iMol)
        #     print("trainsed:", smi)

        imp_val_bool = True
        for atom in iMol.GetAtoms():

            if atom.GetDegree() > 5:
                # print("this symbol forbidden:",atom.GetSymbol(),"\n\n\n\n\n\n\n")
                imp_val_bool = False
                # print("inner:", imp_val_bool)
                # print("wrong_atom:",smi)
                break

        if smi == "F[S](F)(F)(F)(F)F":
            print("F[S](F)(F)(F)(F)F:", imp_val_bool)

        if (iMol.GetNumAtoms() <= 80) and imp_val_bool:
            smi_ret.append(smi)
            sol_ret.append(sol)
            split_set.append(spl)
    print("data_size:", len(smi_ret))

    # shuffled_ix = np.random.permutation(np.arange(len(smi_ret)))
    #
    # smi_ret = np.array(smi_ret)[shuffled_ix]
    # sol_ret = np.array(sol_ret)[shuffled_ix]
    # split_set = np.array(split_set)[shuffled_ix]
    assert len(smi_ret) == len(sol_ret)
    return smi_ret, sol_ret, split_set


def smiles_to_onehot(smi_list):
    def smiles_to_vector(smiles, vocab, max_length):
        while len(smiles) < max_length:
            smiles += " "
        return [vocab.index(str(x)) for x in smiles]

    vocab = np.load('./vocab.npy')
    smi_total = []
    for smi in smi_list:
        smi_onehot = smiles_to_vector(smi, list(vocab), 120)
        smi_total.append(smi_onehot)
    return np.asarray(smi_total)


def convert_to_graph(smiles_list):
    adj = []
    adj_norm = []
    features = []
    maxNumAtoms = 80
    for i in smiles_list:
        # Mol
        # print("i:", i)
        iMol = Chem.MolFromSmiles(i.strip())
        # Adj
        iAdjTmp = rdmolops.GetAdjacencyMatrix(iMol)
        # print("iAdjTmp.shape[0]",iAdjTmp.shape[0])
        # if (iAdjTmp.shape[0] > maxNumAtoms):
        #     print("!!!!!\n!!!!!\n!!!!\n")
        assert iAdjTmp.shape[0] <= maxNumAtoms

        # Feature
        if (iAdjTmp.shape[0] <= maxNumAtoms):
            # Feature-preprocessing
            iFeature = np.zeros((maxNumAtoms, 58))
            iFeatureTmp = []
            for atom in iMol.GetAtoms():
                try:
                    iFeatureTmp.append(atom_feature(atom))  ### atom features only
                except:
                    print("wrong data:", i.strip(), "wrong_atom:", atom.GetSymbol())
            iFeature[0:len(iFeatureTmp), 0:58] = iFeatureTmp  ### 0 padding for feature-set
            features.append(iFeature)

            # Adj-preprocessing
            iAdj = np.zeros((maxNumAtoms, maxNumAtoms))
            iAdj[0:len(iFeatureTmp), 0:len(iFeatureTmp)] = iAdjTmp + np.eye(len(iFeatureTmp))
            adj.append(np.asarray(iAdj))

    features = np.asarray(features)

    return features, adj


def atom_feature(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'H', 'Si', 'P', 'Cl', 'Br',
                                           'Li', 'Na', 'K', 'Mg', 'Ca', 'Fe', 'As', 'Al', 'I', 'B',
                                           'V', 'Tl', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
                                           'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'Mn', 'Cr', 'Pt', 'Hg', 'Pb']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                    [atom.GetIsAromatic()])  # (40, 6, 5, 6, 1)


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    # print list((map(lambda s: x == s, allowable_set)))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def mini_batch(X_train,Y_train,start_ind,batchsize=32):
    if start_ind+batchsize<=len(Y_train):
        return X_train[start_ind:start_ind+batchsize],Y_train[start_ind:start_ind+batchsize],start_ind+batchsize
    else:
        end_ind=(start_ind+batchsize)%len(Y_train)
        X_batch=np.concatenate([X_train[start_ind:len(Y_train)],X_train[:end_ind]],axis=0)
        Y_batch=np.concatenate([Y_train[start_ind:len(Y_train)],Y_train[:end_ind]],axis=0)
        return X_batch,Y_batch,end_ind

