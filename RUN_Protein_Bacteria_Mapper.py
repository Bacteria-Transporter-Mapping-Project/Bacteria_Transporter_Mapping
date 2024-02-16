import os, sys, json
from Protein_Bacteria_Mapper import Protein_Bacteria_Mapper
from StabilitySelection import StabilitySelection
import pandas as pd
from IPython.display import display

disease = sys.argv[1]

data_dir = '~/PK/Priya_et_al'
supp_12_17_excel = os.path.join(data_dir, 'Supplementary Tables S12-S17.xlsx')
re_supp_13_15_17_excel = os.path.join(data_dir, 'REWRITTEN Supplementary Tables S12-S17.xlsx')
patients_excel = os.path.join(data_dir, 'Supplementary Table S1.xlsx')

if disease == 'IBD':
    print('Loading', supp_12_17_excel)
    gene_df = pd.read_excel(supp_12_17_excel, engine='openpyxl', sheet_name='S14', index_col=0).T
    print('Loading', re_supp_13_15_17_excel)
    bact_df = pd.read_excel(re_supp_13_15_17_excel, engine='openpyxl', sheet_name='S15', index_col=[0,1,2,3,4,5,6,7]).T
    print('Loading', patients_excel)
    patients_df = pd.read_excel(patients_excel, engine='openpyxl', sheet_name='IBD_metadata', index_col=0).replace('nonIBD', 'healthy')
    health_profiles = patients_df['Diagnosis'].value_counts().index.to_list()


drug_trsp = ['ABCA1',
 'ABCA10',
 'ABCA12',
 'ABCA13',
 'ABCA2',
 'ABCA3',
 'ABCA4',
 'ABCA5',
 'ABCA6',
 'ABCA7',
 'ABCA8',
 'ABCA9',
 'ABCB1',
 'ABCB10',
 'ABCB11',
 'ABCB4',
 'ABCB5',
 'ABCB6',
 'ABCB7',
 'ABCB8',
 'ABCB9',
 'ABCC1',
 'ABCC10',
 'ABCC11',
 'ABCC12',
 'ABCC2',
 'ABCC3',
 'ABCC4',
 'ABCC5',
 'ABCC6',
 'ABCC8',
 'ABCC9',
 'ABCD1',
 'ABCD2',
 'ABCD3',
 'ABCD4',
 'ABCE1',
 'ABCF1',
 'ABCF2',
 'ABCF3',
 'ABCG1',
 'ABCG2',
 'ABCG4',
 'ABCG5',
 'ABCG8',
 'CYP11A1',
 'CYP11B1',
 'CYP11B2',
 'CYP17A1',
 'CYP19A1',
 'CYP1A1',
 'CYP1A2',
 'CYP1B1',
 'CYP20A1',
 'CYP21A2',
 'CYP24A1',
 'CYP26A1',
 'CYP26B1',
 'CYP26C1',
 'CYP27A1',
 'CYP27B1',
 'CYP27C1',
 'CYP2A13',
 'CYP2A6',
 'CYP2A7',
 'CYP2B6',
 'CYP2C18',
 'CYP2C19',
 'CYP2C8',
 'CYP2C9',
 'CYP2D6',
 'CYP2E1',
 'CYP2F1',
 'CYP2J2',
 'CYP2R1',
 'CYP2S1',
 'CYP2U1',
 'CYP2W1',
 'CYP39A1',
 'CYP3A4',
 'CYP3A43',
 'CYP3A5',
 'CYP3A7',
 'CYP46A1',
 'CYP4A11',
 'CYP4A22',
 'CYP4B1',
 'CYP4F11',
 'CYP4F12',
 'CYP4F2',
 'CYP4F22',
 'CYP4F3',
 'CYP4F8',
 'CYP4V2',
 'CYP4X1',
 'CYP4Z1',
 'CYP51A1',
 'CYP7A1',
 'CYP7B1',
 'CYP8B1',
 'SLC10A1',
 'SLC10A2',
 'SLC10A3',
 'SLC10A4',
 'SLC10A5',
 'SLC10A6',
 'SLC10A7',
 'SLC11A1',
 'SLC11A2',
 'SLC12A1',
 'SLC12A2',
 'SLC12A3',
 'SLC12A4',
 'SLC12A5',
 'SLC12A6',
 'SLC12A7',
 'SLC12A8',
 'SLC12A9',
 'SLC13A1',
 'SLC13A2',
 'SLC13A3',
 'SLC13A4',
 'SLC13A5',
 'SLC14A1',
 'SLC14A2',
 'SLC15A1',
 'SLC15A2',
 'SLC15A3',
 'SLC15A4',
 'SLC15A5',
 'SLC16A1',
 'SLC16A10',
 'SLC16A11',
 'SLC16A12',
 'SLC16A13',
 'SLC16A14',
 'SLC16A2',
 'SLC16A3',
 'SLC16A4',
 'SLC16A5',
 'SLC16A6',
 'SLC16A7',
 'SLC16A8',
 'SLC16A9',
 'SLC17A1',
 'SLC17A2',
 'SLC17A3',
 'SLC17A4',
 'SLC17A5',
 'SLC17A6',
 'SLC17A7',
 'SLC17A8',
 'SLC17A9',
 'SLC18A1',
 'SLC18A2',
 'SLC18A3',
 'SLC18B1',
 'SLC19A1',
 'SLC19A2',
 'SLC19A3',
 'SLC1A1',
 'SLC1A2',
 'SLC1A3',
 'SLC1A4',
 'SLC1A5',
 'SLC1A6',
 'SLC1A7',
 'SLC20A1',
 'SLC20A2',
 'SLC22A1',
 'SLC22A10',
 'SLC22A11',
 'SLC22A12',
 'SLC22A13',
 'SLC22A14',
 'SLC22A15',
 'SLC22A16',
 'SLC22A17',
 'SLC22A18',
 'SLC22A18AS',
 'SLC22A2',
 'SLC22A23',
 'SLC22A24',
 'SLC22A25',
 'SLC22A3',
 'SLC22A31',
 'SLC22A4',
 'SLC22A5',
 'SLC22A6',
 'SLC22A7',
 'SLC22A8',
 'SLC22A9',
 'SLC23A1',
 'SLC23A2',
 'SLC23A3',
 'SLC24A1',
 'SLC24A2',
 'SLC24A3',
 'SLC24A4',
 'SLC24A5',
 'SLC25A1',
 'SLC25A10',
 'SLC25A11',
 'SLC25A12',
 'SLC25A13',
 'SLC25A14',
 'SLC25A15',
 'SLC25A16',
 'SLC25A17',
 'SLC25A18',
 'SLC25A19',
 'SLC25A2',
 'SLC25A20',
 'SLC25A21',
 'SLC25A22',
 'SLC25A23',
 'SLC25A24',
 'SLC25A25',
 'SLC25A26',
 'SLC25A27',
 'SLC25A28',
 'SLC25A29',
 'SLC25A3',
 'SLC25A30',
 'SLC25A31',
 'SLC25A32',
 'SLC25A33',
 'SLC25A34',
 'SLC25A35',
 'SLC25A36',
 'SLC25A37',
 'SLC25A38',
 'SLC25A39',
 'SLC25A4',
 'SLC25A40',
 'SLC25A41',
 'SLC25A42',
 'SLC25A43',
 'SLC25A44',
 'SLC25A45',
 'SLC25A46',
 'SLC25A47',
 'SLC25A48',
 'SLC25A5',
 'SLC25A51',
 'SLC25A52',
 'SLC25A53',
 'SLC25A6',
 'SLC26A1',
 'SLC26A10',
 'SLC26A11',
 'SLC26A2',
 'SLC26A3',
 'SLC26A4',
 'SLC26A5',
 'SLC26A6',
 'SLC26A7',
 'SLC26A8',
 'SLC26A9',
 'SLC27A1',
 'SLC27A2',
 'SLC27A3',
 'SLC27A4',
 'SLC27A5',
 'SLC27A6',
 'SLC28A1',
 'SLC28A2',
 'SLC28A3',
 'SLC29A1',
 'SLC29A2',
 'SLC29A3',
 'SLC29A4',
 'SLC2A1',
 'SLC2A10',
 'SLC2A11',
 'SLC2A12',
 'SLC2A13',
 'SLC2A14',
 'SLC2A2',
 'SLC2A3',
 'SLC2A4',
 'SLC2A4RG',
 'SLC2A5',
 'SLC2A6',
 'SLC2A7',
 'SLC2A8',
 'SLC2A9',
 'SLC30A1',
 'SLC30A10',
 'SLC30A2',
 'SLC30A3',
 'SLC30A4',
 'SLC30A5',
 'SLC30A6',
 'SLC30A7',
 'SLC30A8',
 'SLC30A9',
 'SLC31A1',
 'SLC31A2',
 'SLC32A1',
 'SLC33A1',
 'SLC34A1',
 'SLC34A2',
 'SLC34A3',
 'SLC35A1',
 'SLC35A2',
 'SLC35A3',
 'SLC35A4',
 'SLC35A5',
 'SLC35B1',
 'SLC35B2',
 'SLC35B3',
 'SLC35B4',
 'SLC35C1',
 'SLC35C2',
 'SLC35D1',
 'SLC35D2',
 'SLC35D3',
 'SLC35E1',
 'SLC35E2B',
 'SLC35E3',
 'SLC35E4',
 'SLC35F1',
 'SLC35F2',
 'SLC35F3',
 'SLC35F4',
 'SLC35F5',
 'SLC35F6',
 'SLC35G1',
 'SLC35G2',
 'SLC35G3',
 'SLC35G5',
 'SLC35G6',
 'SLC36A1',
 'SLC36A2',
 'SLC36A3',
 'SLC36A4',
 'SLC37A1',
 'SLC37A2',
 'SLC37A3',
 'SLC37A4',
 'SLC38A1',
 'SLC38A10',
 'SLC38A11',
 'SLC38A2',
 'SLC38A3',
 'SLC38A4',
 'SLC38A5',
 'SLC38A6',
 'SLC38A7',
 'SLC38A8',
 'SLC38A9',
 'SLC39A1',
 'SLC39A10',
 'SLC39A11',
 'SLC39A12',
 'SLC39A13',
 'SLC39A14',
 'SLC39A2',
 'SLC39A3',
 'SLC39A4',
 'SLC39A5',
 'SLC39A6',
 'SLC39A7',
 'SLC39A8',
 'SLC39A9',
 'SLC3A1',
 'SLC3A2',
 'SLC40A1',
 'SLC41A1',
 'SLC41A2',
 'SLC41A3',
 'SLC43A1',
 'SLC43A2',
 'SLC43A3',
 'SLC44A1',
 'SLC44A2',
 'SLC44A3',
 'SLC44A4',
 'SLC44A5',
 'SLC45A1',
 'SLC45A2',
 'SLC45A3',
 'SLC45A4',
 'SLC46A1',
 'SLC46A2',
 'SLC46A3',
 'SLC47A1',
 'SLC47A2',
 'SLC48A1',
 'SLC4A1',
 'SLC4A10',
 'SLC4A11',
 'SLC4A1AP',
 'SLC4A2',
 'SLC4A3',
 'SLC4A4',
 'SLC4A5',
 'SLC4A7',
 'SLC4A8',
 'SLC4A9',
 'SLC50A1',
 'SLC51A',
 'SLC51B',
 'SLC52A1',
 'SLC52A2',
 'SLC52A3',
 'SLC5A1',
 'SLC5A10',
 'SLC5A11',
 'SLC5A12',
 'SLC5A2',
 'SLC5A3',
 'SLC5A4',
 'SLC5A5',
 'SLC5A6',
 'SLC5A7',
 'SLC5A8',
 'SLC5A9',
 'SLC6A1',
 'SLC6A11',
 'SLC6A12',
 'SLC6A13',
 'SLC6A14',
 'SLC6A15',
 'SLC6A16',
 'SLC6A17',
 'SLC6A18',
 'SLC6A19',
 'SLC6A2',
 'SLC6A20',
 'SLC6A3',
 'SLC6A4',
 'SLC6A5',
 'SLC6A6',
 'SLC6A7',
 'SLC6A8',
 'SLC6A9',
 'SLC7A1',
 'SLC7A10',
 'SLC7A11',
 'SLC7A13',
 'SLC7A14',
 'SLC7A2',
 'SLC7A3',
 'SLC7A4',
 'SLC7A5',
 'SLC7A6',
 'SLC7A6OS',
 'SLC7A7',
 'SLC7A8',
 'SLC7A9',
 'SLC8A1',
 'SLC8A2',
 'SLC8A3',
 'SLC8B1',
 'SLC9A1',
 'SLC9A2',
 'SLC9A3',
 'SLC9A3R1',
 'SLC9A3R2',
 'SLC9A4',
 'SLC9A5',
 'SLC9A6',
 'SLC9A7',
 'SLC9A8',
 'SLC9A9',
 'SLC9B1',
 'SLC9B2',
 'SLC9C1',
 'SLC9C2',
 'SLCO1A2',
 'SLCO1B1',
 'SLCO1B3',
 'SLCO1B7',
 'SLCO1C1',
 'SLCO2A1',
 'SLCO2B1',
 'SLCO3A1',
 'SLCO4A1',
 'SLCO4C1',
 'SLCO5A1',
 'SLCO6A1'] 
ss_n_iters = 100

# Test
# ss_n_iters = 1
# gene_df = gene_df.iloc[:,:2]
# bact_df = bact_df.iloc[:,:2]
# drug_trsp = ['A1BG', 'A1CF']

mapper = Protein_Bacteria_Mapper(gene_exp=gene_df, 
                                 bact_exp=bact_df, 
                                 patients=patients_df,
                                 genes=drug_trsp,
                                 health_profiles=health_profiles)

mapper.run(fdr_tresh=0.1)

output_dir = '~/PK'


json.dump(mapper.validated_bact_inds_dict, open('validated_bact_inds.json', 'w'))
json.dump(mapper.validated_bact_dict, open('validated_bact.json', 'w'))
json.dump(mapper.lasso_bact_dict, open('lasso_bact.json', 'w'))
json.dump(mapper.lasso_bact_inds_dict, open('lass_bact_inds.json', 'w'))
json.dump(mapper.fdr_bact_dict, open('fdr_bact.json', 'w'))
json.dump(mapper.fdr_bact_inds_dict, open('fdr_bact_inds.json', 'w'))

import numpy as np
np.save('coef.npy', mapper.coef)

ss = StabilitySelection(X=mapper.X,
                        y=mapper.y,
                        lambda_best=mapper.alpha)

ss.run(n_iters=ss_n_iters, fwer_thresh=0.1)


json.dump(ss.stability_bact_dict, open('stability_bact.json', 'w'))
json.dump(ss.stability_bact_inds_dict, open('stability_bact_inds.json', 'w'))
