## Data Source

The C-MAPSS dataset can be downloaded from NASA's CoE at:
```
https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
```
The raw files are not redistributed in this repository.


### Raw files expected in the C-MAPSS dataset

The pipelines in this repository expect cleaned CMAPSS files w/ column headers generated from the NASA raw text files using the three simple steps below.

Expected raw files (FD001–FD004):

FD001 - ```train_FD001.txt, test_FD001.txt, RUL_FD001.txt```

FD002 - ```train_FD002.txt, test_FD002.txt, RUL_FD002.txt```

FD003 - ```train_FD003.txt, test_FD003.txt, RUL_FD003.txt```

FD004 - ```train_FD004.txt, test_FD004.txt, RUL_FD004.txt```


### Expected input files for experiments

The input files for experiments are provided in ``` data/processed/ ``` and the expected files for each subset are:

FD001:
- ```train_FD001_cleaned.csv, test_FD001_cleaned, rul_FD001.csv```

FD002:
- ```train_FD002_cleaned.csv, test_FD002_cleaned.csv, rul_FD002.csv```

FD003:
- ```train_FD003_cleaned.csv, test_FD003_cleaned.csv, rul_FD003.csv```

FD004:
- ```train_FD004_cleaned.csv, test_FD004_cleaned.csv, rul_FD004.csv```
---

## Cleaning Steps

#### 1. Column Headers

The raw C-MAPSS text files do not include a header row. This project assigns the following column names in order:

- ```engine_id, cycle, op_setting1, op_setting2, op_setting3, sensor1, sensor2, ..., sensor21```

#### 2. Operating settings (by subset):

- Single operating condition subsets (FD001/FD003): **drop** `op_setting1`, `op_setting2`, `op_setting3`

- Multi-condition subsets (FD002/FD004): **keep** `op_setting1`, `op_setting2`, `op_setting3`

#### 3. Sensors removed (all subsets)

The following sensors are removed from all subsets:

- ```sensor1, sensor5, sensor10, sensor16, sensor18, sensor19```

#### 4. Engine_id column added to RUL file

For each subset, the engine_id column is added to its corresponding True RUL for the test set.

#### Processed dataset location

After preprocessing, processed files should be written to:
```
data/processed/
```
