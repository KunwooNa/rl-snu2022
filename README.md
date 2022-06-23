# Description 
This repository is the sourse code for the team project of "Data Science and Reinforcement Learning" course in SNU. 
* Following codes are provided by the instructor. 
    - chain_mdp.py
    - lava_grid.py
    - test.py
    - interaction_chainMDP.py
    - interaction_lava.py

# How to implement test.py

#### Arguments
* --team : your team number
* --envType : 0 for chain mdp, 1 for lava grid
* --evalType : 0 for evaluating performance, 1 for evaluation sample efficiency
* --seeds : the list of seeds

#### Examples
* chain mdp & performance
    - python test.py --team 1 --envType 0 --evalType 0 --seeds 1 10 100 1000 10000   
* chain mdp & sample efficiency
    - python test.py --team 1 --envType 0 --evalType 1 --seeds 1 10 100 1000 10000
* lava grid & Performance
    - python test.py --team 1 --envType 1 --evalType 0 --seeds 1 10 100 1000 10000
* lava grid & sample efficiency
    - python test.py --team 1 --envType 1 --evalType 1 --seeds 1 10 100 1000 10000

#### output file
* For chain-mdp, chain-pf.txt(or chain-se.txt) will be created for performance (or sample efficiency).
* For lava-grid, lava-pf.txt(or lava-se.txt) will be created for performance (or sample efficiency).
