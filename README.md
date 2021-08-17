# Policy Gradient algorithms

## REINFORCE

Implementation of the REINFORCE algorithm as described in the paper Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning [[1]](#references).

The algorithm follows REINFORCE with Baseline (episodic) from Reinforcement Learning: An Introduction, p. 330 [[2]](#references).

## A2C

Implementation of the A2C algorithm.

## DPG

Implementation of the DPG algorithm.

I provide a `main.py` as well as a Jupyter Notebook which demonstrate how to set up, train and compare multiple agents.

## Project Structure

    ├── README.md
    ├── main.py                             # Lab where agents are declared, trained and compared
    ├── .gitignore
    ├── agents
    │   ├──  reinforce_agent.py             # REINFORCE agent
    │   ├──  a2c_agent.py                   # A2C agent
    │   └──  dpg_agent.py                   # DPG agent
    └── utils
        ├── network_architectures.py        # A collection of network architectures for policy, value and action-value approximation
        ├── wrappers.py                     # Wrappers and utilities to create Gym environments
        └── plot.py                         # Plot utilities to display agents' performances

In `wrappers.py`, I also provide a clean implementation of a CartPole Swing Up environment. The pole starts hanging down and the cart must first swing the pole to an upright position before balancing it as in normal CartPole.

## Instructions

First download the source code.
```
git clone https://github.com/maxencefaldor/policy-gradient.git
```
Finally setup the environment and install policy-gradient's dependencies
```
pip install -U pip
pip install -r policy-gradient/requirements.txt
```

### Requirements

- [PyTorch](http://pytorch.org/)
- [NumPy](https://numpy.org/)
- [Gym](https://gym.openai.com/)
- [atari-py](https://github.com/openai/atari-py)
- [Matplotlib](https://matplotlib.org/)

## Acknowledgements

- [@pytorch](https://github.com/pytorch) for [Reinforce-PyTorch](https://github.com/hagerrady13/Reinforce-PyTorch)
- [@hagerrady13](https://github.com/hagerrady13) for [reinforce.py](https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py)

## References

[1] [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://people.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf), Ronald J. Williams, 1992.  
[2] [Reinforcement Learning: An Introduction](http://www.incompleteideas.net/sutton/book/ebook/the-book.html), Sutton and Barto, 1998.  
[3] [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf), Sutton et al., 1999.  
[4] [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783), Mnih et al., 2016.  
[5] [Deterministic Policy Gradient Algorithms](http://proceedings.mlr.press/v32/silver14.pdf), Silver et al., 2014.  
[6] [Distributed Distributional Deterministic Policy Gradients](https://arxiv.org/abs/1804.08617), Hoffman et al., 2018.  
