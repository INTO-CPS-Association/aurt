aurt --logger-config .\resources\logging.conf compile-rbd --mdh .\resources\robot_parameters\twolink_dh.csv --out rigid_body_dynamics
if ($LASTEXITCODE -ne 0)
{
    exit $LASTEXITCODE
}
aurt --logger-config .\resources\logging.conf compile-rd --model-rbd rigid_body_dynamics --friction-torque-model square --friction-viscous-powers 2 1 4 --out robot_dynamics
if ($LASTEXITCODE -ne 0)
{
    exit $LASTEXITCODE
}
aurt --logger-config .\resources\logging.conf calibrate --model robot_dynamics --data .\resources\Dataset\ur5e_45degX_aurt_demo_1\ur5e_45degX_aurt_demo_1.csv --gravity 0.0 6.937 -6.937 --out-params calibrated_parameters.csv --out-calibrated-model rd_calibrated
if ($LASTEXITCODE -ne 0)
{
    exit $LASTEXITCODE
}
aurt --logger-config .\resources\logging.conf calibrate-validate --model robot_dynamics --data .\resources\Dataset\ur5e_45degX_aurt_demo_1\ur5e_45degX_aurt_demo_1.csv --gravity 0 6.937 -6.937 --calibration-data-rel 0.7 --out-params calibrated_parameters.csv --out-calibrated-model rd_calibrated --out-prediction predicted_output.csv
exit $LASTEXITCODE