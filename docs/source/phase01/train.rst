.. _phase01_train:

Train
=================================

Four models are used as part of the Phase 01 Results Management system, each responsible for one of the tasks listed below.

* determines if there are findings or no findings
* if findings are found, determine if there are lung or adrenal findings, 
* if findings are found, determine the relevant portion of the note that made that decision, 
* and, if there are lung findings, determine if a chest CT is recommended.

The functions to train these models are provided in the file ``nmrezman.phase01.train.general.py`` and named as follows.

* :func:`nmrezman.phase01.train.general.train_findings_model`
* :func:`nmrezman.phase01.train.general.train_lung_adrenal_model`
* :func:`nmrezman.phase01.train.general.train_comment_model`
* :func:`nmrezman.phase01.train.general.train_lung_recommended_proc_model`



Findings vs No Finding Model
-------------------------------------------------

This model classifies whether the report contains findings or no findings. This is the first model the report is run through. This biLSTM model uses GloVe word embeddings.
Training was run via the script:

.. code-block:: bash

    python -m nmrezman.phase01.train.train_findings --data_path /path/to/data/reports_df.gz --glove_embedding_path /path/to/data/glove.6B.300d.txt --model_checkpoint_name /path/to/results/phase01/findings/findings_best_model.h5 --result_fname /path/to/results/phase01/findings/findings_best_result.log --tokenizer_fname /path/to/results/phase01/findings/tokenizer.gz


.. autofunction:: nmrezman.phase01.train.general.train_findings_model


Lung vs Adrenal Findings Model
-------------------------------------------------

This model classifies whether the report contains lung or adrenal findings. This model is run if the Findings vs No Findings model identifies findings were found. This biLSTM model uses BioWordVec word embeddings.

Training was run via the script:

.. code-block:: bash

    python -m nmrezman.phase01.train.train_lung_adrenal --data_path /path/to/data/reports_df.gz --bioword_path /path/to/data/BioWordVec_PubMed_MIMICIII_d200.bin --model_checkpoint_name /path/to/results/phase01/lung_adrenal/lung_adrenal_best_model.h5 --result_fname /path/to/results/phase01/lung_adrenal/lung_adrenal_best_result.log --tokenizer_fname /path/to/results/phase01/findings/tokenizer.gz


.. autofunction:: nmrezman.phase01.train.general.train_lung_adrenal_model


Comment Extraction Model
-------------------------------------------------

This model classifies the comment in the report that indicate the relevant finding. This model is run if the Findings vs No Findings model identifies findings were found. This is an XGBoost-based model.

Training was run via the script:

.. code-block:: bash

    python -m nmrezman.phase01.train.train_comment --data_path /path/to/data/reports_df.gz --model_checkpoint_name /path/to/results/phase01/comment/comment_best_model.sav --result_fname /path/to/results/phase01/comment/comment_best_result.log


.. autofunction:: nmrezman.phase01.train.general.train_comment_model


Lung Recommended Procedure Model
-------------------------------------------------

This model classifies the comment in the report that indicate the relevant finding. This model is run if the Lung vs Adrenal Findings model identifies lung findings were found. This is a biLSTM model.

Training was run via the script:

.. code-block:: bash

    python -m nmrezman.phase01.train.train_lung_recommended_proc_model --data_path /path/to/data/reports_df.gz --model_checkpoint_name /path/to/results/phase01/lung_recommend/lung_recommend_best_model.h5 --result_fname /path/to/results/phase01/lung_recommend/lung_recommend_best_result.log --tokenizer_fname /path/to/results/phase01/findings/tokenizer.gz


.. autofunction:: nmrezman.phase01.train.general.train_lung_recommended_proc_model
