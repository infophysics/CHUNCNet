import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import csv

files = [
    'cmssm_higgs_dm_lsp_sigma.csv',
    'cmssm_higgs_dm_lsp_sigma2.csv',
    'cmssm_higgs_dm_lsp_tophat.csv',
    'cmssm_higgs_dm_lsp_tophat2.csv',
    'pmssm_higgs_dm_lsp_no_gap_sigma.csv',
    'pmssm_higgs_dm_lsp_no_gap_sigma2.csv',
    'pmssm_higgs_dm_lsp_no_gap_tophat.csv',
    'pmssm_higgs_dm_lsp_no_gap_tophat2.csv',
]

names = [
    r"CMSSM $\sigma=0.0001$",
    r"CMSSM $\sigma=0.001$",
    r"CMSSM $\lambda=0.0001$",
    r"CMSSM $\lambda=0.001$",
    r"PMSSM $\sigma=0.0001$",
    r"PMSSM $\sigma=0.001$",
    r"PMSSM $\lambda=0.0001$",
    r"PMSSM $\lambda=0.001$",
]
ticks = [ii for ii in range(len(names))]

colors = [
    'k', 'r', 'c', 'm'
]

fig, axs = plt.subplots(figsize=(10,6))

for ii, file in enumerate(files):
    higgs, dm, higgs_dm, higgs_dm_lsp = [], [], [], []
    with open(file, "r") as f:
        reader = csv.reader(f, delimiter=",")
        next(reader)
        for row in reader:
            higgs.append(float(row[2])/float(row[1]))
            dm.append(float(row[3])/float(row[1]))
            higgs_dm.append(float(row[4])/float(row[1]))
            higgs_dm_lsp.append(float(row[5])/float(row[1]))
    axs.errorbar(
        ii, np.mean(higgs), 
        yerr=np.std(higgs)/np.sqrt(len(higgs)),
        capsize=2,
        c=colors[0]
    )
    axs.errorbar(
        ii, np.mean(dm), 
        yerr=np.std(dm)/np.sqrt(len(dm)),
        capsize=2, 
        c=colors[1]
    )
    axs.errorbar(
        ii, np.mean(higgs_dm), 
        yerr=np.std(higgs_dm)/np.sqrt(len(higgs_dm)),
        capsize=2, 
        c=colors[2]
    )
    axs.errorbar(
        ii, np.mean(higgs_dm_lsp), 
        yerr=np.std(higgs_dm_lsp)/np.sqrt(len(higgs_dm_lsp)),
        capsize=2, 
        c=colors[3]
    )
axs.plot([],[],label="Higgs",c=colors[0])
axs.plot([],[],label="Dark Matter",c=colors[1])
axs.plot([],[],label="Higgs/Dark Matter",c=colors[2])
axs.plot([],[],label="Higgs/Dark Matter/LSP",c=colors[3])
axs.set_xticks(ticks)
axs.set_xticklabels(names, rotation=45)
axs.set_ylabel("Average Validity")
plt.title("Average Validity vs. Sampling Method")
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.savefig("results.png")
plt.show()