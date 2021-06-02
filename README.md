# AU Robotics Toolbox (AURT) Overview

# Installation

To install the tool, type:
```
pip install aurt
```

# Command Line Interface

The following shows the different use cases that aurt supports.
Each use case can be further customized in a configuration file.

## Linearize

```
aurt linearize --mdh mdh_file.csv --gravity [0.0, 0.0, 9.81] --out linearized_model.pickle
```
Reads the Modified Denavit Hartenberg (MDH) parameters in file `mdh_file.csv`, and writes the linearized and reduced model to file `linearized_model.pickle`.
The gravity vector determines the orientation of the robot for which the parameters will be calibrated.

## Calibrate

```
aurt calibrate --model linearized_model.pickle --data measured_data.csv --joint-friction- --reduced-params calibrated_parameters.csv
```

Reads the linearized model produced in [Linearize](#linearize), the measured data in `measured_data.csv`, and writes the reduced parameter values to `calibrated_parameters.csv`.

In order to generate the original parameters described in `mdh_file.csv`, provided in [Linearize](#linearize), use the `--full-params calibrated_parameters.csv` instead of `--reduced-params`

The measured data should contain the following fields:
- `time`
- ...


## Predict

```
aurt predict --model linearized_model.pickle --data measured_data.csv --reduced-params calibrated_parameters.csv --prediction prediction.csv
```

Reads the linearized model produced in [Linearize](#linearize), the measured data in `measured_data.csv`, and the reduced parameter values produced in [Calibrate](#calibrate), and writes the prediction to `prediction.csv`.


# Contributing

## Development environment

To setup the development environment:
1. Open terminal in the current folder.
2. Optional: create a virtual environment: `python -m venv venv`
3. Optional: activate the virtual environment: 
   1. Windows (Powershell):`.\venv\Scripts\Activate.ps1`
   2. Linux: `source venv/bin/activate`
4. Install all packages for development: `pip install -e .[dev]`
5. To run all tests, open a powershell in the dynamic folder, and run the `.\build_script.ps1 --run-tests offline` script.
6. Optional: open Pycharm in the current folder.


## Dataset management

### Small dataset (< 100MB compressed)

If the data is small, then:
- Each round of experiments should be placed in a folder with an informative name, inside the Dataset folder.
- There should be a readme file in there explaining the steps to reproduce the experiment, parameters, etc...
- The csv files should be 7ziped and committed. Do not commit the csv file.
- There should be tests that use the data there.

### Large Datasets (>= 100MB compressed)

If the data is large, then:

- A "lite" version of the dataset should be in the dataset folder (following the same guidelines as before)
  - This is important to run the tests.
- the larger version should be placed in the shared drive (see below).

There is a shared drive for large datasets.
The shared drive **Nat_UR-robot-datasets** has been created with **Emil Madsen** as owner.

| **Navn/Name**         | **Drev-ansvarlig/Shared  drive owner** |
| --------------------- | -------------------------------------- |
| Nat_UR-robot-datasets | au504769  (Emil Madsen)                |


 **Read/write access is assigned to:** 

| **Brugernavn/Username** | **Navn/Name**                   | **Afdeling/department** | **E-mail**                                                | **Tilføjet via  gruppe/assigned by group** |
| ----------------------- | ------------------------------- | ----------------------- | --------------------------------------------------------- | ------------------------------------------ |
| au602135                | Cláudio  Ângelo Gonçalves Gomes | Cyber-Physical  Systems | [claudio.gomes@ece.au.dk](mailto:claudio.gomes@ece.au.dk) |                                            |

For more information on access, self-service and management of files: https://medarbejdere.au.dk/en/administration/it/guides/datastorage/data-storage/



# Tasks

- [ ] To discuss:
  - [ ] Do we need an API, or just CLI?
  - [ ] Logging framework
    - [ ] Must be configurable via config file.
  - [ ] What to config (and what to provide as input, and what to allow override)
    - [ ] Logging
    - [ ] friction model (allow override from CLI)
  - [ ] Add some kind of progress indicator to the Linearize process.
  - [ ] Configuration file format. My suggestion: toml. Other possible choices: yaml.
    - [ ] Must support comments
    - [ ] Must integrate with logging
- [ ] Finish data schema

- [x] Gravity vector goes into the config file in the linearize step.
- [ ] Joint dynamic configuration will be hybrid (overridable via the CLI.)
- [ ] In the predict stage, there's two caching mechanisms to build the regressor:
  - [ ] rigid body dynamics, 
  - [ ] joint dynamics
- [ ] When do we given MDH parameters?
