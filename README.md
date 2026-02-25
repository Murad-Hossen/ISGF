## Feature-based morphological analysis of shape graph data
<img width="1240" height="308" alt="pipeline" src="https://github.com/user-attachments/assets/d97cb72f-5c08-4322-bc84-af637c93e2cb" />



## Overview

This work introduces and evaluates a computational pipeline for the statistical analysis of shape graph datasets, namely geometric networks embedded in 2D or 3D spaces. Our proposed approach relies on the extraction of a specifically curated and explicit set of topological, geometric and directional features, designed to satisfy key invariance properties. We leverage the resulting feature representation for tasks such as group comparison, clustering and classification on cohorts of shape graphs. The effectiveness of this representation is evaluated on several real-world datasets including urban road/street networks, neuronal traces and astrocyte imaging. These results are benchmarked against several alternative methods, both feature-based and not.


## Citation

If you use this code or methodology in your research, please cite:

```bibtex
@article{hossen2026feature,
  title={Feature-based morphological analysis of shape graph data},
  author={Hossen, Murad and Labate, Demetrio and Charon, Nicolas},
  journal={arXiv preprint arXiv:2602.16120},
  year={2026}
}
```

Source code link:
```
https://github.com/Murad-Hossen/ISGF.git
```

### Key Capabilities

- **Feature Extraction**: Extract 19 comprehensive features from 3D neuron morphologies, 2D astrocyte morphology, and urban road networks
- **Statistical Comparison**: Perform group-level statistical analysis across different datasets.
- **Clustering Analysis**: Compare networks using both feature-based methods and Gromov-Wasserstein Distance (G-W) with t-SNE visualization.
- **Classification**: Perform shape graph classification using Random Forest classifiers.



---



**Core Dependencies:**
- numpy >= 1.26.4
- scipy >= 1.13.1
- scikit-learn >= 1.5.0
- matplotlib
- networkx
- pandas
- osmnx (for road network analysis)
- porespy (for fractal dimension computation)
- POT (Python Optimal Transport, for GWD computation)

### Python Version
Python 3.10+ recommended

---





---
