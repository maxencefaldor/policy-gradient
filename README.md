# Policy Gradient algorithms

## REINFORCE

Implementation of the REINFORCE algorithm and as described in the paper Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning [[1]](#references).

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
    │   └──  reinforce_agent.py             # REINFORCE agent
    └── utils
        ├── network_architectures.py        # A collection of network architectures including actors without baseline, with baseline and baseline with shared parameters
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
