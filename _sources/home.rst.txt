=================================
NM Results Management AI
=================================

This documentation provides information about the work and `code base <https://github.com/mozzilab/NM_Radiology_AI>`_ for the models and integrated workflow described in `Preventing Delayed and Missed Care by Applying Artificial Intelligence to Trigger Relevant Imaging Follow-Up. <https://catalyst.nejm.org/doi/full/10.1056/CAT.21.0469>`_ Our :ref:`"blog post"<about_blog_post>` provides a condensed version of the paper and allows you to hear from key players (e.g., physicians, program managers, and engineers)! We also provide :ref:`documentation<code>` for our `source code <https://github.com/mozzilab/NM_Radiology_AI>`_. 

The goal of the project was to use natural language processing (NLP) to identify lung or adrenal-related findings that required follow-up. Ultimately, various models were trained and then deployed. As part of the deployment process, automatic alerts are sent to both physicians and patients to schedule and and track completion of follow-ups. See below for a high-level system overview. 

.. figure:: /imgs/solution.svg
   :align: center
   :scale: 66%

   ..
   
In the example below, a lung finding is detected by our models and a chest CT is recommended. Accordingly, the integrated framework would trigger generation of a Best Practice Advisory (BPA) to alert the ordering physician, presenting workflows where follow-up studies can be ordered as appropriate.

.. panels::

   :card: shadow
   :column: col-lg
   :header: bg-secondary text-light

   Sample Radiology Report
   ^^^^^^^^^^^^^^
   
      PROCEDURE:  CT CHEST W CONTRAST. HISTORY:  Colon cancer. TECHNIQUE:  Non-ionic contrast enhanced helical thoracic CT was performed. FINDINGS:   Support Devices:  None. Heart/Pericardium/Great Vessels:        Cardiac size is normal.      There is mild calcific coronary artery atherosclerosis.       There is no pericardial effusion.      The thoracic aorta is normal in diameter.      The main pulmonary artery is normal in diameter. Pleural Spaces:  The pleural spaces are clear. Mediastinum/Hila:  There is no mediastinal or hilar lymph node enlargement. Neck Base/Chest Wall/Diaphragm/Upper Abdomen:  There is no supraclavicular or axillary lymph node enlargement.  Please refer to the same day MRI abdomen/pelvis for full description of findings and the upper abdomen.  No aggressive appearing bone lesions. Lungs/Central Airways:  The central airways are clear.  Incidental note is made of a blind-ending cardiac bronchus arising from the bronchus intermedius.  A 7 mm solid nodule in the left upper lobe has increased in size, previously 5 mm (4/53).  The 2 mm right lower lobe nodule is stable. CONCLUSIONS:   Continued increase in size of the solid left upper lobe nodule now measuring 7 mm.  Given the continued slow growth, this lesion is concerning for metastatic disease.  Consider PET/CT or tissue sampling. &#x20; FINAL REPORT Attending Radiologist: 

   ++++++++++++++
   
   Model Output:

      .. code-block:: rst

         procedure:     Chest CT

         noduleType:    Lung

         followUpFlag:  Findings Present

         followUpText:  continued increase in size of the solid left upper 
                        lobe nodule now measuring 7 mm.  given the continued 
                        slow growth, this lesion is concerning for metastatic 
                        disease.  consider pet/ct or tissue sampling.


.. note::

   Please cite us! 

   .. code-block:: none

      @article{doi:10.1056/CAT.21.0469,
         author = {Jane Domingo  and Galal Galal  and Jonathan Huang  and Priyanka Soni  and Vladislav Mukhin  and Camila Altman  and Tom Bayer  and Thomas Byrd  and Stacey Caron  and Patrick Creamer  and Jewell Gilstrap  and Holly Gwardys  and Charles Hogue  and Kumar Kadiyam  and Michael Massa  and Paul Salamone  and Robert Slavicek  and Michael Suna  and Benjamin Ware  and Stavroula Xinos  and Lawrence Yuen  and Thomas Moran  and Cynthia Barnard  and James G. Adams  and Mozziyar Etemadi },
         title = {Preventing Delayed and Missed Care by Applying Artificial Intelligence to Trigger Radiology Imaging Follow-up},
         journal = {NEJM Catalyst},
         volume = {3},
         number = {4},
         pages = {CAT.21.0469},
         year = {2022},
         doi = {10.1056/CAT.21.0469},

         URL = {https://catalyst.nejm.org/doi/abs/10.1056/CAT.21.0469},
         eprint = {https://catalyst.nejm.org/doi/pdf/10.1056/CAT.21.0469}
         ,
      }



