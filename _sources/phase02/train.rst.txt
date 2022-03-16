.. _phase02_train:

Train
=================================

Four models are used as part of the Phase 02 Results Management system, each responsible for one of the tasks listed below.

* determines if there are lung findings, adrenal findings, or no findings
* if findings are found, determine the relevant portion of the note that made that decision, 
* and, if there are lung findings, determine if a chest CT is recommended.

The functions to train these models are provided in the file ``nmrezman.phase02.train.general.py`` and named as follows.

* :func:`nmrezman.phase01.train.general.train_findings_model`
* :func:`nmrezman.phase01.train.general.train_comment_model`
* :func:`nmrezman.phase01.train.general.train_lung_recommended_proc_model`


Lung Findings, Adrenal Findings, or No Findings Model
------------------------------------------------------

This model classifies whether the report contains lung findings, adrenal findings, or no findings. This is the first model the report is run through. This is an MLM RoBERTa-based model.

Training was run via the script:

.. code-block:: bash

    python -m nmrezman.phase02.train.train_findings --data_path /path/to/df.gz --model_pretrained_path /path/to/pretrained_models --output_dir /path/to/results/findings/ --logging_dir /path/to/findings/logging --result_fname /path/to/findings/findings_best_result.log --wandb_dir /path/to/results/findings_recommend/ --do_reporting True 


.. autofunction:: nmrezman.phase02.train.general.train_findings_model


Comment Extraction Model
-------------------------------------------------

This model classifies the comment in the report that indicate the relevant finding. This model is run if the Findings model identifies findings were found. This is a Question-Answer based model.

Training was run via the script:

.. code-block:: bash

    python -m nmrezman.phase02.train.train_comment --data_path /path/to/df.gz --output_dir /path/to/results/comment/ --result_fname_prefix annotation_label

.. autofunction:: nmrezman.phase02.train.general.train_comment_model


Lung Recommended Procedure Model
-------------------------------------------------

This model classifies whether a Chest CT or some other ("ambiguous") procedure is recommended. This model is run if the Findings model identifies lung findings were found. This is an MLM RoBERTa-based model.

Training was run via the script:

.. code-block:: bash

    python -m nmrezman.phase02.train.train_findings --data_path /path/to/df.gz --model_pretrained_path /path/to/pretrained_models --output_dir /path/to/results/lung_proc/ --logging_dir /path/to/lung_proc/logging --result_fname /path/to/lung_proc/lung_proc_best_result.log --wandb_dir /path/to/results/lung_proc_recommend/ --do_reporting True


.. autofunction:: nmrezman.phase02.train.general.train_lung_recommended_proc_model
