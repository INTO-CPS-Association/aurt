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

To setup the development environment:
1. Open terminal in the current folder.
2. Optional: create a virtual environment: `python -m venv venv`
3. Optional: activate the virtual environment: 
   1. Windows (Powershell):`.\venv\Scripts\Activate.ps1`
   2. Linux: `source venv/bin/activate`
4. Install all packages for development: `pip install -e .[dev]`
5. To run all tests, open a powershell in the dynamic folder, and run the `.\build_script.ps1 --run-tests offline` script.
6. Optional: open Pycharm in the current folder.

# Tasks

- [ ] To discuss:
  - [x] Name tool: aurt
  - [ ] New Repo.
  - [ ] Do we need an API, or just CLI?
  - [ ] Why do we need to read excel?
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
