Prerequisites
=================================

Data
---------------------------------

To train your own models on your own data, you will need labeled data. Trained in annotating radiology reports, NM nurses labeled 36,385 reports with 5,779 (15.9%) and 409 (1.1%) lung and adrenal follow-up recommendations, respectively, for this work. Ultimately, these reports were saved into a dataframe according to the table below. Note that this can be customized to fit your needs: different data sources, classification problems, etc. A de-identified example of such a dataframe is provided below.

+-----------------------+-----------------------+---------------------------------------------------------------+
| Info                  | Dataframe Heading     | Values                                                        |
+=======================+=======================+===============================================================+
| Radiology Report      | ``note``              | Entire report directly from Epic                              |
+-----------------------+-----------------------+---------------------------------------------------------------+
| Finding               | ``selected_finding``  | ``No Findings``, ``Lung Findings``, or ``Adrenal Findings``   |
|                       |                       | :sup:`1`                                                      |
+-----------------------+-----------------------+---------------------------------------------------------------+
| Recommended Procedure | ``selected_proc``     | ``CT Chest``, ``Ambiguous``                                   |
|                       |                       | :sup:`1,2`                                                    |
+-----------------------+-----------------------+---------------------------------------------------------------+
| Relevant Portion of   | ``selected_label``    | Findings portion of the report                                |
| the Report with       |                       | :sup:`1`                                                      |
| respect to the        |                       |                                                               |
| Finding               |                       |                                                               |
+-----------------------+-----------------------+---------------------------------------------------------------+
| De-identified report  | ``rpt_num``           | Randomly generated                                            |
| number                |                       |                                                               |
+-----------------------+-----------------------+---------------------------------------------------------------+

| :sup:`1` *Nurse annotated / provided*
| :sup:`2` *Only provided for reports with Lung Findings*

For labeling, we used the open source `INCEpTION platform <https://inception-project.github.io/>`_. A screenshot of the platform is shown below.

.. image:: /imgs/inception.png
   :align: center


.. csv-table:: Sample Radiology Report Dataframe
   :file: demo_data.csv
   :header-rows: 1
   :widths: 5, 80, 5, 5, 5


.. note::

    For Phase 01 of the project, all preprocessing of the note (i.e., extracting the findings, removing Dr. signatures, etc.) was completed once and saved off as part of the dataframe in the column as ``new_note``. Note that the Phase 01 source code will make use of this column, while the Phase 02 code will reference the original ``note`` and perform the preprocessing at the beginning of the training functions.



Required Downloads
---------------------------------

Some of the model training scripts require downloads from the internet. Please find them below. Note that the Phase 02 models use 🤗 (Hugging Face), which will download the files automatically.

+---------------+-----------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------+
| Project Phase | Model                                                     | Downloads                                                                                                                                                     | Download Beforehand?  |
+===============+===========================================================+===============================================================================================================================================================+=======================+
| Phase 01      | Findings vs No Finding Model                              | `GloVe pretrained word vectors: glove.6B.300d.txt <https://nlp.stanford.edu/data/glove.6B.zip>`_                                                              | |chk|                 |
|               +-----------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------+
|               | Lung vs Adrenal Findings Model                            | `BioWordVec word vectors: BioWordVec_PubMed_MIMICIII_d200.bin <https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioWordVec_PubMed_MIMICIII_d200.bin>`_    | |chk|                 |
|               +-----------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------+
|               | Comment Extraction Model                                  | `NLTK Punkt Tokenizer <https://www.nltk.org/api/nltk.tokenize.punkt.html#module-nltk.tokenize.punkt>`_                                                        | |crs|                 |
|               +-----------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------+
|               | Lung Recommended Procedure Model                          | N/A                                                                                                                                                           | N/A                   |
+---------------+-----------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------+
| Phase 02      | Lung Findings, Adrenal Findings, or No Findings Model     | `RoBERTa base <https://huggingface.co/roberta-base>`_                                                                                                         | |crs|                 |
|               +-----------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------+
|               | Comment Extraction Model                                  | `RoBERTa base <https://huggingface.co/roberta-base>`_                                                                                                         | |crs|                 |
|               +-----------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------+
|               | Lung Recommended Procedure Model                          | `RoBERTa large <https://huggingface.co/roberta-large>`_                                                                                                       | |crs|                 |
+---------------+-----------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------+

.. note::

    You must have these tools downloaded before running the Phase 01 models. Do this by either manual downloading and extracting in the workspace or using ``wget`` in the cli to accomplish the same.


.. |chk|   unicode:: U+02713 .. CHECK MARK
.. |crs|   unicode:: U+2717 .. CROSS MARK