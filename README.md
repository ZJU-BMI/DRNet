DRNet
==
This is the code for implementing DRNet proposed in "On learning disentangled representations for individual treatment effect estimation" (under review). 

It is written in python 3.7 with numpy 1.16.2 and tensorflow 1.13.1.

The code of DRNet is built upon the Counterfactual regression (CFR) work of Johansson, Shalit & Sontag (2016) and Shalit, Johansson & Sontag (2016), https://github.com/clinicalml/cfrnet.
The parameter searching, network training and evaluation follow the procedures of CFR to ensure fair comparison.

To run parameter search:

```python drnet_param_search.py <config_file> <num_runs>```

To evaluate the results:

```python evaluate.py <config_file> [overwrite] [filters]```

Dataset
-
Both IHDP and Jobs dataset are avaliable from "https://www.fredjo.com/"

Example
-
To run parameter search:
```python drnet_param_search.py configs/example_ihdp_DRNet.txt 10```

To evaluate the results:
```python evaluate.py configs/example_ihdp_DRNet.txt 1```
