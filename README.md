# Is the Graph Attention Network Effective for Multivariate Time Series Anomaly Detection?

## SMD Preprocessing
- The experimental dataset for this study, the Server Machine Dataset (SMD), should be imported from the link below and placed in the path 'datasets/ServerMachineDataset'.
  - link: https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset
  - Since we only have one example, we can only see the results of one experiment at this time.
- Once you have placed the dataset, you will need to preprocess the SMD using the commands below.
```
python preprocess.py --dataset SMD
```


## Ablation Study
- Ablation studies can be done by looking at 'Ablation_Test_Code.ipynb'.
- Specifically, 'lib.py' contains the basic code provided in the MTAD-GAT paper, and 'MTAD_GAT_ABLATION.py' contains the model implemented for ablation and code that automatically shows the training and prediction results when you specify the path to the dataset and the ablation range to the model.
