.. _code:

Code Overview
=================================

Code Summary
---------------------------------

The code is split into two main sections: Phase 01 and Phase 02. Phase 01 models include stacked biLSTM models used in the first phase of deployment. Phase 02 models include transformer models pretrained on NM radiology reports and later finetuned for various tasks in the pipeline. The block diagram below shows the different models for each phase of the project.

.. note:: 

   Phase 01 refers to the original models deployed. Phase 02 refers to the updated models, which were refined after the initial clinical deployment with the aim of improving scalability and model performance. Moreover, we leveraged the latest advances for deep learning NLP. With respect to machine learning frameworks, Phase 01 utilizes the Tensorflow and Keras libraries, while Phase 02 leverages the |hf| (Hugging Face) platform.

.. |hf|   unicode:: U+1F917 .. HUGGING FACE

.. image:: /imgs/models.svg
   :align: center
   :scale: 100%

.. panels::
    :card: shadow


    .. link-button:: phase01/phase01
        :type: ref
        :text: Phase 01: BiLSTM and XGBoost Models
        :classes: btn-secondary btn-block
    
    ^^^
    Explore the source code for :ref:`training<phase01_train>` the models and using them to :ref:`classify<phase01_classify>` reports. These were the initial models used.

    ---
    .. link-button:: phase02/phase02
        :type: ref
        :text: Phase 02: MLM Models
        :classes: btn-secondary btn-block

    ^^^
        Explore the source code for :ref:`training<phase02_train>` the new models and using them to :ref:`classify<phase02_classify>` reports. These models are currently implemented at NM as part of a Result Management system.


Code Organization
---------------------------------

The code is organized as shown below (condensed such that python ``__init__.py`` files, etc. are not included). Note that files ``train/train_**.py`` are ease-of-use scripts, which train models that are defined in ``train/general.py``. Likewise, the ``run_classifier.py`` files are scripts to easily classify raw radiology report text when provided with trained model weights.

::

   src/
   ├─nmrezman/
   │ ├─phase01/
   │ │ ├─classify/
   │ │ │ ├─classifier.py
   │ │ │ └─run_classifier.py
   │ │ ├─train/
   │ │ │ ├─general.py
   │ │ │ ├─train_comment.py
   │ │ │ ├─train_findings.py
   │ │ │ ├─train_lung_adrenal.py
   │ │ │ └─train_lung_recommended_proc.py
   │ │ └─models.py
   │ ├─phase02/
   │ │ ├─classify/
   │ │ │ ├─classifier.py
   │ │ │ └─run_classifier.py
   │ │ └─train/
   │ │   ├─general.py
   │ │   ├─pretrain.py
   │ │   ├─train_comment.py
   │ │   ├─train_findings.py
   │ │   └─train_lung_recommended_proc.py
   │ └─utils.py
   └─setup.py


Using This Code 
-------------------------------------------------

This documentation provides the source code used to train all models, which can be modified to fit your needs. There are a several different ways you could go about this.

1.  Training can be run from a cloned repo by running the script as a module. For example, to train the Phase 01 Findings vs No Findings model, use the command:

   .. code-block:: bash

      cd src
      python -m nmrezman.phase01.train.train_findings --data_path /path/to/data/reports_df.gz --glove_embedding_path /path/to/data/glove.6B.300d.txt --model_checkpoint_name /path/to/results/phase01/findings/findings_best_model.h5 --result_fname /path/to/results/phase01/findings/findings_best_result.log --tokenizer_fname /path/to/results/phase01/findings/tokenizer.gz

2. Directly run the scripts or import the functions into python once :obj:`nmrezman` has been pip installed as a python package from either `GitHub <https://github.com/mozzilab/NM_Radiology_AI>`_ directly or, if the repo is cloned locally, from the local directory. See the commands below.

   .. code-block:: bash

      pip install "git+https://github.com/mozzilab/NM_Radiology_AI.git@main#egg=nmrezman"

   or if the repo is already installed locally

   .. code-block:: bash

      pip install /path/to/repo/NM_Radiology_AI

   Once pip installed, the training functions can be imported directly.

   .. code-block:: python

      from nmrezman.phase01.train.general import train_findings_model

      result = train_findings_model(        
            data_path="/path/to/data/reports_df.gz",
            glove_embedding_path="/path/to/data/glove.6B.300d.txt",
            model_checkpoint_name="/path/to/results/phase01/findings/findings_best_model.h5",
            result_fname="/path/to/results/phase01/findings/findings_best_result.log",
            tokenizer_fname="/path/to/results/phase01/findings/tokenizer.gz",
      )

3. Last but not least, use the pre-built container with everything packaged in, ready to go. The image contains the complete environment used to build these models as well as a click-through walkthrough to get you started. The source code for the container image is available in our `github repo <https://github.com/mozzilab/NM_Radiology_AI/docker>`_, and the pre-built image is publicly available on our docker-hub repository, `mozzilab/nmrezman <https://hub.docker.com/repository/docker/mozzilab/nmrezman>`_.

   The only requirements are that ``docker`` is installed and all the required drivers are up to date.

   GPU command (suggested):
 
   .. code-block:: shell

      docker run -it --rm --net=host -e ip_addr=${IP_ADDR} --ulimit memlock=-1 --gpus all mozzilab/nmrezman:latest
   
   CPU command (suggested):

   .. code-block:: shell

      docker run -it --rm --net=host -e ip_addr=${IP_ADDR} mozzilab/nmrezman:latest

   Required args:
      * ``-it`` - opens an interactive tty, effectively it just takes you straight to the cmd line inside the container
      * ``-net=host`` - binds the host computer's network to the container, so all ports are inherently exposed. You can specify specific ports for Jupyter and code server by including the port binding(s) in the format ``-p 8081:8081`` along the environmental variable(s) ``-e VSCODE_PORT=8081`` & ``-e JUPYTER_PORT=8889``
      * ``mozzilab/nmrezman:latest`` - name of the container image
      * ``--gpus all`` - if using GPU(s) to run the model, you must include this flag
      * ``--ulimit memlock=-1`` - prevent the locking of shared memory, need if running GPUs

   Optional args:
      * ``--mount type=bind,src=${PATH_TO_DATA},dst=/workspace/data`` - bind a folder into the ``/workspace/data`` folder in the container.
      * ``--rm`` - sets container to be ephemeral, so all resources are disposed of upon the container being stopped.
      * ``-e ip_addr=${IP_ADDR}`` - only if operating on remote machine, will make the ip:port auto-print message work nicely (ctrl-click).
      
.. warning:: 
   The code will likely need to be modified to suit your needs (at a minimum, preprocessing raw reports and dataframe structuring). Generalizability of this code to other health care systems is not guaranteed and only reflects the 10 hospitals and one electronic medical record for which it was tested. However, modifying the code (e.g., preprocessing, base model checkpoints, model constants) may yield similar results.