Overview
=================================

A block diagram of the Phase 01 models is provided below. The :ref:`demo notebook<Demo: Findings vs No Findings Model>` provides an annotated version of the source code for one of the models: the Findings vs No Findings model. Since many of the models in Phase 01 use this same format, this should provide a good example for understanding how the models during this phase of the project were trained. To view the actual source code (``*.py`` files) used for training and classifying, please refer to the :ref:`train<phase01_train>` and :ref:`classify<phase01_classify>` documentation.

.. image:: /imgs/phase01.svg
   :align: center
   :scale: 85%
   

In the table below, we provide a description of the different tasks and architectures of the models. 

+--------------------------+--------------------------------------+--------------------------+--------------------------------------------------------------------------+
| Model Name               | Task                                 | Architecture             | Source Code Function                                                     | 
+==========================+======================================+==========================+==========================================================================+ 
| Findings or No Findings  | Identify if the report has (i) lung  | Stacked BiLSTM           | :func:`nmrezman.phase01.train.general.train_findings_model`              |
|                          | or adrneal findings or (ii) no       | (Tensorflow / Keras)     |                                                                          | 
|                          | findings                             |                          |                                                                          | 
+--------------------------+--------------------------------------+--------------------------+--------------------------------------------------------------------------+ 
| Lung or Adrenal Findings | Identify if the report has (i) lung  | Stacked BiLSTM           | :func:`nmrezman.phase01.train.general.train_lung_adrenal_model`          |
|                          | or (ii) adrneal findings             | (Tensorflow / Keras)     |                                                                          | 
|                          |                                      |                          |                                                                          | 
+--------------------------+--------------------------------------+--------------------------+--------------------------------------------------------------------------+ 
| Lung Recommended         | Identify if a report with lung has   | Stacked BiLSTM           | :func:`nmrezman.phase01.train.general.train_lung_recommended_proc_model` |
| Procedure (i.e., Chest   | findings recommends (i) a Chest CT   | (Tensorflow / Keras)     |                                                                          | 
| CT or Ambiguous Follow-  | follow-up or (ii) some other         |                          |                                                                          | 
| up Recommended)          | ambiguous procedure                  |                          |                                                                          | 
+--------------------------+--------------------------------------+--------------------------+--------------------------------------------------------------------------+ 
| Comment Extraction       | Identify the portion of the report   | XGBoost                  | :func:`nmrezman.phase01.train.general.train_comment_model`               |
|                          | that contains the findings text      |                          |                                                                          | 
+--------------------------+--------------------------------------+--------------------------+--------------------------------------------------------------------------+ 
