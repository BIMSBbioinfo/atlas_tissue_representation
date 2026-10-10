**Figure. Test performance of the supervised VAE as a function of training-set size.**
The supervised VAE (flexynesis `supervised_vae`, fixed hyperparameters, one HPO iteration) was trained on random subsamples
of 10%, 20%, ..., 100% of the training set (11,800 to 118,000 of 118,263 samples) and evaluated on the same held-out test set
in every case. Uberon tissue classification performance is shown as **(A)** weighted-average F1 score, **(B)** balanced accuracy and **(C)** Cohen's kappa. **(D)** Spearman correlation between measured and reconstructed gene expression on the test set, computed per sample (correlation across genes; blue) and per gene (correlation across samples; orange), summarized as the median over samples or genes. Points show the mean over 10 independent seeds and error
bars show ±1 standard deviation across seeds. Subsamples were drawn at random without stratification, so rare tissues may be
under-represented at small sizes. All models used batch size 1024.
