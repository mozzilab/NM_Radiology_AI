Overview
=================================

A block diagram of the Phase 02 models is provided below. Two of the models use the similar fine-tuning methods, and as such, we provide a :ref:`demo notebook for fine-tuning<Demo: Fine-Tuning NM Results Management Language Model with a Custom Dataset>`, providing extended comments on the code. Additionally, since we employ pretraining before fine-tuning, we include a :ref:`demo notebook for pretraining<Demo: Pretraining the NM Results Management Language Model with Custom Corpus>`. To view the actual source code (``*.py`` files) used for training and classifying, please refer to the :ref:`train<phase02_train>` and :ref:`classify<phase02_classify>` documentation.

.. image:: /imgs/phase02.svg
   :align: center
   :scale: 85%

In the table below, we provide a description of the different tasks and architectures of the models. 

+--------------------------+--------------------------------------+--------------------------+--------------------------------------------------------------------------+
| Model Name               | Task                                 | Architecture             | Source Code Function                                                     | 
+==========================+======================================+==========================+==========================================================================+ 
| Lung Findings, Adrenal   | Identify if the report has (i) lung  | MLM based on RoBERTa     | :func:`nmrezman.phase02.train.general.train_findings_model`              |
| Findings, or No Findings | (ii) adrneal, or (iii) or no         | (|hf|)                   |                                                                          | 
|                          | findings                             |                          |                                                                          | 
+--------------------------+--------------------------------------+--------------------------+--------------------------------------------------------------------------+ 
| Lung Recommended         | Identify if a report with lung has   | MLM based on RoBERTa     | :func:`nmrezman.phase02.train.general.train_lung_recommended_proc_model` |
| Procedure (i.e., Chest   | findings recommends (i) a Chest CT   | (|hf|)                   |                                                                          |  
| CT or Ambiguous Follow-  | follow-up or (ii) some other         |                          |                                                                          | 
| up Recommended)          | ambiguous procedure                  |                          |                                                                          | 
+--------------------------+--------------------------------------+--------------------------+--------------------------------------------------------------------------+ 
| Comment Extraction       | Identify the portion of the report   | Question-Answer based    | :func:`nmrezman.phase02.train.general.train_comment_model`               |
|                          | that contains the findings text      | RoBERTa                  |                                                                          | 
|                          |                                      | (``simpletransformers``  |                                                                          | 
|                          |                                      | via |hf|)                |                                                                          | 
+--------------------------+--------------------------------------+--------------------------+--------------------------------------------------------------------------+ 

.. |hf|   unicode:: U+1F917 .. HUGGING FACE