| a                              | b                                    |   n_seeds |   mean_diff |   ci95_lo |   ci95_hi |   cohens_dz |   p_corrected | a_better   |   p_holm |
|:-------------------------------|:-------------------------------------|----------:|------------:|----------:|----------:|------------:|--------------:|:-----------|---------:|
| Fine-tuned attn_gate [photo12] | EfficientNet-B0 (frozen, image only) |        10 |      0.0197 |    0.0128 |    0.0266 |      3.6256 |        0.0001 | 10/10      |   0.0014 |
| Fine-tuned attn_gate [photo12] | XGBoost [12 shortcut-free]           |        10 |      0.0028 |   -0.0019 |    0.0074 |      0.7481 |        0.2148 | 8/10       |   0.6445 |
| Fine-tuned attn_gate [photo12] | Gated attention [12 shortcut-free]   |        10 |      0.0005 |   -0.0051 |    0.0060 |      0.1081 |        0.8514 | 5/10       |   0.8514 |
| Fine-tuned concat [photo12]    | EfficientNet-B0 (frozen, image only) |        10 |      0.0219 |    0.0137 |    0.0301 |      3.3913 |        0.0002 | 10/10      |   0.0021 |
| Fine-tuned concat [photo12]    | XGBoost [12 shortcut-free]           |        10 |      0.0050 |    0.0021 |    0.0079 |      2.1834 |        0.0036 | 10/10      |   0.0328 |
| Fine-tuned concat [photo12]    | Gated attention [12 shortcut-free]   |        10 |      0.0027 |   -0.0008 |    0.0062 |      0.9802 |        0.1143 | 8/10       |   0.6206 |
| Fine-tuned img_gap [photo12]   | EfficientNet-B0 (frozen, image only) |        10 |      0.0130 |    0.0080 |    0.0180 |      3.2834 |        0.0002 | 10/10      |   0.0024 |
| Fine-tuned img_gap [photo12]   | XGBoost [12 shortcut-free]           |        10 |     -0.0039 |   -0.0105 |    0.0027 |     -0.7451 |        0.2165 | 3/10       |   0.6445 |
| Fine-tuned img_gap [photo12]   | Gated attention [12 shortcut-free]   |        10 |     -0.0062 |   -0.0141 |    0.0017 |     -0.9903 |        0.1111 | 1/10       |   0.6206 |
| Fine-tuned attn_gate [photo12] | Fine-tuned img_gap [photo12]         |        10 |      0.0067 |    0.0010 |    0.0123 |      1.4871 |        0.0264 | 10/10      |   0.1846 |
| Fine-tuned concat [photo12]    | Fine-tuned img_gap [photo12]         |        10 |      0.0089 |    0.0032 |    0.0146 |      1.9852 |        0.0063 | 10/10      |   0.0504 |
| Fine-tuned attn_gate [photo12] | Fine-tuned concat [photo12]          |        10 |     -0.0022 |   -0.0050 |    0.0006 |     -1.0157 |        0.1034 | 1/10       |   0.6206 |