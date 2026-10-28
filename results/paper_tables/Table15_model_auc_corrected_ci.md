| cohort                            | model                                        |   mean_auc |     sd |   ci95_lo |   ci95_hi |
|:----------------------------------|:---------------------------------------------|-----------:|-------:|----------:|----------:|
| QSO vs galaxy (matched)           | EfficientNet-B0 (frozen, image only)         |     0.9716 | 0.0063 |    0.9636 |    0.9796 |
| QSO vs galaxy (matched)           | Gated attention [12 shortcut-free]           |     0.9908 | 0.0032 |    0.9867 |    0.9949 |
| QSO vs galaxy (matched)           | XGBoost [12 shortcut-free]                   |     0.9886 | 0.0027 |    0.9851 |    0.9920 |
| QSO vs galaxy (matched)           | Logistic regression [colours]                |     0.9547 | 0.0067 |    0.9462 |    0.9631 |
| QSO vs galaxy (matched)           | Colour-cut selector (u-g)                    |     0.5661 | 0.0169 |    0.5447 |    0.5875 |
| QSO vs galaxy (matched)           | Morphology rule (concentration)              |     0.6634 | 0.0236 |    0.6335 |    0.6933 |
| QSO vs galaxy (matched)           | Logistic regression [14]                     |     0.9962 | 0.0031 |    0.9923 |    1.0000 |
| QSO vs galaxy (matched)           | Random forest [14]                           |     0.9989 | 0.0015 |    0.9970 |    1.0000 |
| QSO vs galaxy (matched)           | SVM (RBF) [14]                               |     0.9981 | 0.0023 |    0.9952 |    1.0000 |
| QSO vs galaxy (matched)           | XGBoost [14]                                 |     0.9985 | 0.0020 |    0.9960 |    1.0000 |
| QSO vs galaxy (matched)           | MLP [14]                                     |     0.9970 | 0.0026 |    0.9937 |    1.0000 |
| QSO vs galaxy (matched)           | MLP [12 shortcut-free]                       |     0.9808 | 0.0042 |    0.9755 |    0.9861 |
| QSO vs galaxy (matched)           | XGB | Full tabular features                  |     0.9985 | 0.0020 |    0.9960 |    1.0000 |
| QSO vs galaxy (matched)           | MLP | Full tabular features                  |     0.9966 | 0.0030 |    0.9928 |    1.0000 |
| QSO vs galaxy (matched)           | Fusion | Full tabular features               |     0.9980 | 0.0018 |    0.9957 |    1.0000 |
| QSO vs galaxy (matched)           | XGB | Without redshift                       |     0.9984 | 0.0017 |    0.9963 |    1.0000 |
| QSO vs galaxy (matched)           | MLP | Without redshift                       |     0.9955 | 0.0028 |    0.9920 |    0.9991 |
| QSO vs galaxy (matched)           | Fusion | Without redshift                    |     0.9979 | 0.0019 |    0.9954 |    1.0000 |
| QSO vs galaxy (matched)           | XGB | Without luminosity proxy               |     0.9985 | 0.0020 |    0.9960 |    1.0000 |
| QSO vs galaxy (matched)           | MLP | Without luminosity proxy               |     0.9950 | 0.0035 |    0.9906 |    0.9994 |
| QSO vs galaxy (matched)           | Fusion | Without luminosity proxy            |     0.9971 | 0.0019 |    0.9946 |    0.9995 |
| QSO vs galaxy (matched)           | XGB | Without redshift + luminosity proxy    |     0.9886 | 0.0027 |    0.9851 |    0.9920 |
| QSO vs galaxy (matched)           | MLP | Without redshift + luminosity proxy    |     0.9754 | 0.0048 |    0.9693 |    0.9815 |
| QSO vs galaxy (matched)           | Fusion | Without redshift + luminosity proxy |     0.9908 | 0.0032 |    0.9867 |    0.9949 |
| QSO vs galaxy (matched)           | XGB | Photometry + colours only              |     0.9860 | 0.0030 |    0.9822 |    0.9898 |
| QSO vs galaxy (matched)           | MLP | Photometry + colours only              |     0.9722 | 0.0055 |    0.9653 |    0.9791 |
| QSO vs galaxy (matched)           | Fusion | Photometry + colours only           |     0.9914 | 0.0019 |    0.9890 |    0.9939 |
| QSO vs galaxy (matched)           | XGB | Morphology proxies only                |     0.7398 | 0.0190 |    0.7158 |    0.7639 |
| QSO vs galaxy (matched)           | MLP | Morphology proxies only                |     0.7445 | 0.0253 |    0.7123 |    0.7766 |
| QSO vs galaxy (matched)           | Fusion | Morphology proxies only             |     0.9783 | 0.0046 |    0.9724 |    0.9841 |
| QSO vs galaxy (matched)           | Concatenation [12 shortcut-free]             |     0.9927 | 0.0023 |    0.9898 |    0.9956 |
| QSO vs galaxy (matched)           | Cross-attention, no gate [12 shortcut-free]  |     0.9919 | 0.0023 |    0.9890 |    0.9948 |
| QSO vs galaxy (matched)           | Global-token + gate [12 shortcut-free]       |     0.9910 | 0.0022 |    0.9881 |    0.9938 |
| QSO vs galaxy (matched)           | Concatenation [14]                           |     0.9980 | 0.0017 |    0.9958 |    1.0000 |
| QSO vs galaxy (matched)           | Cross-attention, no gate [14]                |     0.9978 | 0.0019 |    0.9954 |    1.0000 |
| QSO vs galaxy (matched)           | Gated attention [14]                         |     0.9980 | 0.0018 |    0.9957 |    1.0000 |
| BPT AGN vs star-forming (matched) | EfficientNet-B0 (frozen, image only)         |     0.8674 | 0.0122 |    0.8519 |    0.8828 |
| BPT AGN vs star-forming (matched) | Concatenation [12]                           |     0.9275 | 0.0067 |    0.9190 |    0.9360 |
| BPT AGN vs star-forming (matched) | Cross-attention, no gate [12]                |     0.9295 | 0.0072 |    0.9204 |    0.9386 |
| BPT AGN vs star-forming (matched) | Global-token + gate [12]                     |     0.9258 | 0.0102 |    0.9129 |    0.9386 |
| BPT AGN vs star-forming (matched) | Gated attention [12]                         |     0.9307 | 0.0076 |    0.9210 |    0.9403 |
| BPT AGN vs star-forming (matched) | Concatenation [14]                           |     0.9303 | 0.0043 |    0.9249 |    0.9358 |
| BPT AGN vs star-forming (matched) | Gated attention [14]                         |     0.9330 | 0.0045 |    0.9273 |    0.9388 |
| BPT AGN vs star-forming (matched) | XGBoost [12]                                 |     0.9224 | 0.0049 |    0.9162 |    0.9285 |
| BPT AGN vs star-forming (matched) | XGBoost [14]                                 |     0.9284 | 0.0036 |    0.9239 |    0.9330 |
| BPT AGN vs star-forming (matched) | XGBoost [colours]                            |     0.9131 | 0.0066 |    0.9048 |    0.9215 |
| BPT AGN vs star-forming (matched) | MLP [12]                                     |     0.9187 | 0.0131 |    0.9022 |    0.9353 |
| BPT AGN vs star-forming (matched) | Colour logistic regression                   |     0.8979 | 0.0068 |    0.8892 |    0.9066 |