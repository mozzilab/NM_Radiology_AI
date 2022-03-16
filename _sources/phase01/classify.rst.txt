.. _phase01_classify:

Classify
=================================

To classify reports using the Phase 01 models, :func:`nmrezman.phase01.classify.classifier` is used. This function takes a raw report, preprocesses the note, loads the model weights, and 

* determines if there are findings or no findings
* if findings are found, determines if there are lung or adrenal findings, 
* if findings are found, determines the relevant portion of the note that made that decision, 
* and, if there are lung findings, determines if a chest CT is recommended.

A script ``nmrezman.phase01.classify.run_classifier`` is provided to easily run report text through the classifier:

.. code-block:: bash

    python -m nmrezman.phase01.classify.run_classifier --data_path /path/to/data.txt --model_path /workspace/local/phase01_platform/model_checkpoints/


.. autofunction:: nmrezman.phase01.classify.classifier