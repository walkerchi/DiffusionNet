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