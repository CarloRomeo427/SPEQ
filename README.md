# SPEQ: Offline Stabilization Phases for Efficient Q-Learning 🚀
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](http://arxiv.org/abs/2501.08669)

Welcome to the official repository for **SPEQ**, a novel reinforcement learning algorithm designed to optimize computational efficiency while maintaining high sample efficiency.

## Overview

SPEQ addresses the challenge of computational inefficiency in high update-to-data (UTD) reinforcement learning methods. It strategically combines low-UTD online training with periodic high-UTD offline stabilization phases. This approach significantly reduces unnecessary gradient computations and improves overall training efficiency.

## Key Benefits

- 🕒 **Reduced Training Time:** Decreases total training time by up to 78%.
- 🌱 **Computationally Efficient:** Performs up to 99% fewer gradient updates.
- 📊 **Improved Performance:** Maintains competitive results compared to state-of-the-art methods.
- 🔄 **Structured Updates:** Periodic offline stabilization phases mitigate overfitting and enhance learning stability.

## Results

Empirical evaluations demonstrate SPEQ's ability to achieve state-of-the-art performance with fewer computational resources. Detailed results and comparisons are available in our published paper.

## Getting Started

Follow these instructions to set up and run SPEQ:

```bash
git clone https://github.com/CarloRomeo427/SPEQ.git
cd SPEQ
pip install -r requirements.txt
python train_speq.py
```

## Future Work

Future enhancements include automatic determination of the timing and duration of stabilization phases, further optimizing computational efficiency.

## Acknowledgements

This implementation is inspired by the work on Dropout Q-Functions by Takuya Hiraoka et al., available at https://github.com/TakuyaHiraoka/Dropout-Q-Functions-for-Doubly-Efficient-Reinforcement-Learning.

## Citation
```bibtex
@article{romeo2025speq,
  title={SPEQ: Offline Stabilization Phases for Efficient Q-Learning in High Update-To-Data Ratio Reinforcement Learning},
  author={Romeo, Carlo and Macaluso, Girolamo and Sestini, Alessandro and Bagdanov, Andrew D},
  journal={arXiv preprint arXiv:2501.08669},
  year={2025}
}
```

Happy training! 🚀
