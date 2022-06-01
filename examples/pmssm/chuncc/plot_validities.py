"""
Plot validities from runs
"""
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import csv

sigma = []
total = []
higgs = []
dm = []
higgs_dm = []
higgs_dm_lsp = []

with open("pmssm_higgs_dm_lsp_no_gap_validities.csv", "r") as file:
    reader = csv.reader(file, delimiter=",")
    next(reader)
    for row in reader:
        total.append(int(row[0]))
        higgs.append(float(row[1])/float(row[0]))
        dm.append(float(row[2])/float(row[0]))
        higgs_dm.append(float(row[3])/float(row[0]))
        higgs_dm_lsp.append(float(row[4])/float(row[0]))
total = np.array(total)
higgs_mean = np.mean(higgs)
dm_mean =np.mean(dm)
higgs_dm_mean = np.mean(higgs_dm)
higgs_dm_lsp_mean = np.mean(higgs_dm_lsp)
higgs_std = np.std(higgs)
dm_std =np.std(dm)
higgs_dm_std = np.std(higgs_dm)
higgs_dm_lsp_std = np.std(higgs_dm_lsp)

print(higgs_mean, higgs_std, dm_mean, dm_std, higgs_dm_mean, higgs_dm_std, higgs_dm_lsp_mean, higgs_dm_lsp_std)

# fig, axs = plt.subplots(figsize=(10,6))
# axs.errorbar(higgs_mean, yerr=higgs_std, label="higgs", capsize=2)
# axs.errorbar(dm_mean, yerr=dm_std, label="dm", capsize=2)
# axs.errorbar(higgs_dm_mean, yerr=higgs_dm_std, label="higgs_dm", capsize=2)
# axs.errorbar(higgs_dm_lsp_mean, yerr=higgs_dm_lsp_std, label="higgs_dm_lsp", capsize=2)
# axs.set_xlabel(r"Latent Sigma ($\sigma$)")
# axs.set_ylabel("Average Validity")
# axs.set_yscale("log")
# plt.legend()
# plt.tight_layout()
# plt.show()
