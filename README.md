# DiffusionNet
## Structure
    src-+-graph.py
        +-module.py
        +-trainer.py
        +-ic_demo.py

## Usage
### fast start
```
    python src/trainer.py
```

### customize
- data set


  in trainer.py
    ```
    datasets={"dataset_name":dgl.data.DataSet}
    ```
- graph processor(currently independent cascade)


    in graph.py
    inherit from GraphProcessor

- model


    in model.py
    inherit from Module

## experiment
- ### cora dataset

|                   model                   | avg_time  | avg_f1-mic |
| :---------------------------------------: | :-------: | :--------: |
|                 GraphSAGE                 |   3.499   |   0.638    |
|                 LightGCN                  |   2.290   |   0.683    |
|            MLP 1 hidden layer             | **1.275** |   0.748    |
|            MLP 2 hidden layers            |   1.996   |   0.723    |
|      LightGCN(**topo not changed**)       |   2.265   |   0.557    |
| MLP 1 hidden layer(**topo not changed**)  |   1.504   | **0.761**  |
| MLP 2 hidden layers(**topo not changed**) |   2.403   |   0.740    |
