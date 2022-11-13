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

with open("pmssm_higgs_dm_lsp_no_gap_validities_tophat.csv", "r") as file:
    reader = csv.reader(file, delimiter=",")
    next(reader)
    for row in reader:
        sigma.append(float(row[0]))
        total.append(int(row[1]))
        higgs.append(float(row[2])/float(row[1]))
        dm.append(float(row[3])/float(row[1]))
        higgs_dm.append(float(row[4])/float(row[1]))
        higgs_dm_lsp.append(float(row[5])/float(row[1]))
sigma = np.array(sigma)
total = np.array(total)
higgs = np.array(higgs)
dm =np.array(dm)
higgs_dm = np.array(higgs_dm)
higgs_dm_lsp = np.array(higgs_dm_lsp)

unique_sigma = np.unique(sigma)
higgs_mean = [np.mean(higgs[(sigma == s)]) for s in unique_sigma]
dm_mean = [np.mean(dm[(sigma == s)]) for s in unique_sigma]
higgs_dm_mean = [np.mean(higgs_dm[(sigma == s)]) for s in unique_sigma]
higgs_dm_lsp_mean = [np.mean(higgs_dm_lsp[(sigma == s)]) for s in unique_sigma]

higgs_std = [np.std(higgs[(sigma == s)]) for s in unique_sigma]
dm_std = [np.std(dm[(sigma == s)]) for s in unique_sigma]
higgs_dm_std = [np.std(higgs_dm[(sigma == s)]) for s in unique_sigma]
higgs_dm_lsp_std = [np.std(higgs_dm_lsp[(sigma == s)]) for s in unique_sigma]

for ii, sigma in enumerate(unique_sigma):
    print(
        sigma,round(higgs_mean[ii],5),round(higgs_std[ii],5),round(dm_mean[ii],5),round(dm_std[ii],5),round(higgs_dm_mean[ii],5),round(higgs_dm_std[ii],5),round(higgs_dm_lsp_mean[ii],5),round(higgs_dm_lsp_std[ii],5)
    )

fig, axs = plt.subplots(figsize=(10,6))
axs.errorbar(unique_sigma, higgs_mean, yerr=higgs_std, label="higgs", capsize=2)
axs.errorbar(unique_sigma, dm_mean, yerr=dm_std, label="dm", capsize=2)
axs.errorbar(unique_sigma, higgs_dm_mean, yerr=higgs_dm_std, label="higgs_dm", capsize=2)
axs.errorbar(unique_sigma, higgs_dm_lsp_mean, yerr=higgs_dm_lsp_std, label="higgs_dm_lsp", capsize=2)
axs.set_xlabel(r"Latent Sigma ($\sigma$)")
axs.set_ylabel("Average Validity")
axs.set_yscale("log")
plt.legend()
plt.tight_layout()
plt.show()
