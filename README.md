## Feature-based statistical analysis of shape graph data



A comprehensive framework for extracting, analyzing, and comparing structural features between biological neural networks (neurons and astrocytes) and urban road networks using graph-theoretic and geometric approaches.

## Overview

ISGA -investigates the structural similarities between biological neural networks and urban road networks by computing a rich set of topological, geometric, and directional features. This project demonstrates that despite operating at vastly different scales and contexts, these networks share remarkable structural commonalities that can be quantified and analyzed using machine learning techniques.


## Citation

If you use this code or methodology in your research, please cite:

```bibtex


```

### Key Capabilities

- **Multi-Domain Feature Extraction**: Extract 19 comprehensive features from 3D neuron morphologies, 2D astrocyte networks, and urban road networks
- **Statistical Comparison**: Perform group-level statistical analysis across different network categories
- **Clustering Analysis**: Compare networks using both feature-based methods and Gromov-Wasserstein Distance (G-W) with t-SNE visualization
- **Classification**: Distinguish network types using Random Forest classifiers



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
