Using the Tool
==============



Compile Rigid Body dynamics
---------------------------

.. code-block:: console

   $ aurt compile-rbd --mdh mdh.csv --gravity 0.0 0.0 -9.81 --out rigid_body_dynamics


Compile Robot Dynamics Model
----------------------------

.. code-block:: console

   $ aurt compile-rd --model-rbd rigid_body_dynamics --friction-load-model square --friction-viscous-powers 2 1 4 --out robot_dynamics


Calibrate
---------

.. code-block:: console

    $ aurt calibrate --model robot_dynamics --data measured_data.csv --out-params calibrated_parameters.csv --out-calibration-model robot_calibration --plot


Predict
-------

.. code-block:: console

    $ aurt predict --model robot_calibration --data measured_data.csv --prediction predicted_output.csv


