## Data source

The C-MAPSS dataset must be downloaded from NASA PCoE (not redistributed in this repository).

### Inputs expected by this repository

The pipelines in this repository expect \*\*cleaned CMAPSS files\*\* generated from the NASA raw text files using the three simple steps below.

Expected raw filenames (FD001–FD004):

FD001 - train\_FD001.txt, test\_FD001.txt, RUL\_FD001.txt

FD002 - train\_FD002.txt, test\_FD002.txt, RUL\_FD002.txt

FD003 - train\_FD003.txt, test\_FD003.txt, RUL\_FD003.txt

FD004 - train\_FD004.txt, test\_FD004.txt, RUL\_FD004.txt
  

### Cleaning Steps

#### 1. Column Headers

The raw C-MAPSS text files do not include a header row. This project assigns the following column names in order:

- engine\_id, cycle, op\_setting1, op\_setting2, op\_setting3, sensor1, sensor2, ..., sensor21

#### 2. Operating settings (by subset):

- Single operating condition subsets (FD001/FD003): drop `op\_setting1`, `op\_setting2`, `op\_setting3`

- Multi-condition subsets (FD002/FD004): keep `op\_setting1`, `op\_setting2`, `op\_setting3`

#### 3. Sensors removed (all subsets)

The following sensors are removed from all subsets:

- sensor1, sensor5, sensor10, sensor16, sensor18, sensor19

### Processed dataset location

After preprocessing, processed files should be written to:

data/processed/

