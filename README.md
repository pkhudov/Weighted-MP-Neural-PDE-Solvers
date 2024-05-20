# Weighted Message Passing Neural PDE Solvers

Follow-up on the work of Johannes Brandstetter*, Daniel Worrall*, Max Welling: [Message Passing Neural PDE Solvers](https://arxiv.org/abs/XXXX.XXXXX). This project builds upon their original repository: [Link to code](https://github.com/original-repo-link).

This repository contains experiments that evaluate the effect of incorporating a Gaussian smoothing kernel for weighting the messages from the neighbor nodes according to their proximity. The main changes are in `experiments/models_gnn.py`.
