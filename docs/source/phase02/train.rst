.. _phase02_train:

Train
=================================

Three models are used as part of the Phase 02 Results Management system, each responsible for one of the tasks listed below.

* determines if there are lung findings, adrenal findings, or no findings
* if findings are found, determine the relevant portion of the note that made that decision, 
* and, if there are lung findings, determine if a chest CT is recommended.

The functions to train these models are provided in the file ``nmrezman.phase02.train.general.py`` and named as follows.

* :func:`nmrezman.phase02.train.general.train_findings_model`
* :func:`nmrezman.phase02.train.general.train_comment_model`
* :func:`nmrezman.phase02.train.general.train_lung_recommended_proc_model`

Before training these models, pretraining was performed via :func:`nmrezman.phase02.train.general.pretrain_roberta_base`.


Pretraining RoBERTa Base Model
------------------------------------------------------

As a first step, we pretrain a DistilRoBERTa base model using radiology reports.

Training was run via the script:

.. code-block:: bash

    python -m nmrezman.phase02.train.pretrain --data_path /path/to/data/reports_df.gz --output_dir /path/to/results/phase02/pretrain --logging_dir /path/to/results/phase02/pretrain/logging --wandb_dir /path/to/results/phase02/pretrain --do_reporting True

.. autofunction:: nmrezman.phase02.train.general.pretrain_roberta_base


Lung Findings, Adrenal Findings, or No Findings Model
------------------------------------------------------

This model classifies whether the report contains lung findings, adrenal findings, or no findings. This is the first model the report is run through. This is an MLM RoBERTa-based model.

Training was run via the script:

.. code-block:: bash

    python -m nmrezman.phase02.train.train_findings --data_path /path/to/data/reports_df.gz --model_pretrained_path /path/to/results/phase02/pretrain/checkpoint-XXXXX --output_dir /path/to/results/phase02/findings/ --logging_dir /path/to/results/phase02/findings/logging --result_fname /path/to/results/phase02/findings/findings_best_result.log --wandb_dir /path/to/results/phase02/findings/findings_recommend/ --do_reporting True 


.. autofunction:: nmrezman.phase02.train.general.train_findings_model


Comment Extraction Model
-------------------------------------------------

This model classifies the comment in the report that indicate the relevant finding. This model is run if the Findings model identifies findings were found. This is a Question-Answer based model.

Training was run via the script:

.. code-block:: bash

    python -m nmrezman.phase02.train.train_comment --data_path /path/to/data/reports_df.gz --output_dir /path/to/results/phase02/comment/ --result_fname_prefix results

.. autofunction:: nmrezman.phase02.train.general.train_comment_model


Lung Recommended Procedure Model
-------------------------------------------------

This model classifies whether a Chest CT or some other ("ambiguous") procedure is recommended. This model is run if the Findings model identifies lung findings were found. This is an MLM RoBERTa-based model.

Training was run via the script:

.. code-block:: bash

    python -m nmrezman.phase02.train.train_findings --data_path /path/to/data/reports_df.gz --model_pretrained_path /path/to/results/phase02/pretrain/checkpoint-XXXXX --output_dir /path/to/results/phase02/lung_recommend/ --logging_dir /path/to/results/phase02/lung_recommend/logging --result_fname /path/to/results/phase02/lung_recommend/lung_recommend_best_result.log --wandb_dir /path/to/results/phase02/lung_recommend/ --do_reporting True


.. autofunction:: nmrezman.phase02.train.general.train_lung_recommended_proc_model
