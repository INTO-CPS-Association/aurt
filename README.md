# AU Robotics Toolbox (AURT) Overview

# Installation

To install the tool, type:
```
pip install aurt
```

# Command Line Interface

The following shows the different use cases that aurt supports.
In order to improve performance, the model is compiled in different stages, 
in a way that allows the user to try alternative friction models without having to re-create the full model, 
which takes a long time.

## Compile Rigid Body Dynamics Model

```
aurt compile-rbd --mdh mdh_file.csv --gravity [0.0, 0.0, 9.81] --out model_rbd.pickle
```
Reads the Modified Denavit Hartenberg (MDH) parameters in file `mdh_file.csv`, and writes the linearized and reduced model to file `linearized_model.pickle`.
The gravity vector determines the orientation of the robot for which the parameters will be calibrated.
The generated model does not include the joint dynamics.

## Complete Joint Dynamics Model

```
aurt compile-jointd --model-rbd model_rbd.pickle --friction-load-model square --friction-viscous-powers [1, 3] --friction-hysteresis-model sign --out model_complete.pickle
```

Reads the rigid body dynamics model created with the previous command, and generates the complete model, 
taking into account the joint dynamics configuration.

The friction configuration options are:
- `--friction-load-model TYPE` where `TYPE in {none, square, absolute}` and:
  - `TYPE=none` means TODO
  - `TYPE=square` means TODO
  - `TYPE=absolute` means TODO 
- `--friction-viscous-powers POWERS` where `POWERS` has the format `[P1, P2, ..., PN]`, and `PN` is a positive real number representing the `N`-th power of the polynomial.
- `--friction-hysteresis-model TYPE` where `TYPE in {sign, maxwell-slip}`, and:
  - `TYPE=sign` means TODO
  - `TYPE=maxwell-slip` means TODO
  
## Calibration

```
aurt calibrate --model model_complete.pickle --data measured_data.csv --out-reduced-params calibrated_parameters.csv
```

Reads the model produced in [Complete Joint Dynamics Model](#complete-joint-dynamics-model), the measured data in `measured_data.csv`, 
and writes the reduced parameter values to `calibrated_parameters.csv`,

In order to generate the original parameters described in `mdh_file.csv`, 
provided in [Compile Rigid Body Dynamics Model](#compile-rigid-body-dynamics-model), 
use the `--out-full-params calibrated_parameters.csv` instead of `--reduced-params`

The measured data should contain the following fields:
- `time`
- TODO

## Predict

```
aurt predict --model model_complete.pickle --data measured_data.csv --reduced-params calibrated_parameters.csv --prediction prediction.csv
```

Reads the model produced in [Complete Joint Dynamics Model](#complete-joint-dynamics-model), 
the measured data in `measured_data.csv`, 
and the reduced parameter values produced in [Calibration](#calibration), and writes the prediction to `prediction.csv`.

The prediction fields are:
- `time`
- TODO

# Contributing

## Development environment

To setup the development environment:
1. Open terminal in the current folder.
2. Optional: create a virtual environment: `python -m venv venv`
3. Optional: activate the virtual environment: 
   1. Windows (Powershell):`.\venv\Scripts\Activate.ps1`
   2. Linux: `source venv/bin/activate`
4. Install all packages for development: `pip install -e .[dev]`
5. Unpack the datasets (see [Dataset management](#dataset-management))
6. To run all tests, open a powershell in the dynamic folder, and run the `.\build_script.ps1 --run-tests offline` script.
7. Optional: open Pycharm in the current folder.


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
- [ ] Finish documentation on friction models
- [x] Gravity vector goes into the config file in the linearize step.
- [ ] Joint dynamic configuration will be hybrid (overridable via the CLI.)
- [ ] In the predict stage, there's two caching mechanisms to build the regressor:
  - [ ] rigid body dynamics, 
  - [ ] joint dynamics
- [ ] When do we given MDH parameters?
- [ ] Make tests load files using project_root() (except linearization_tests.py)
