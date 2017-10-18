# Doctor-patient questions (French)

These are the test and training data used for experiments presented in BioNLP 2017.

## Licence

The data are only aimed for research, educational and non-commercial purposes.

## How to cite

If you use these data, please cite our contribution to BioNLP 2017 as follows:

   [Automatic classification of doctor-patient questions for a virtual patient record query task](http://www.aclweb.org/anthology/W17-2343)  
Leonardo Campillos Llanos, Sophie Rosset, Pierre Zweigenbaum   
*Proc. of BioNLP 2017*, August 4 2017, Vancouver, Canada, pp. 333-341   

  ```
  @inproceedings{Campillos:BIONLP2017,   
  title       = {Automatic classification of doctor-patient questions for a virtual patient record query task},  
  author       = {Campillos Llanos, Leonardo and Rosset, Sophie and Zweigenbaum, Pierre},   
  booktitle = {BioNLP 2017},
  publisher =     {Association for Computational Linguistics},
  location =     {Vancouver, Canada},
  pages     = {333--341},
  year      = 2017,
  urlALT       = {https://doi.org/10.18653/v1/W17-2343.pdf},
  url       = {http://www.aclweb.org/anthology/W17-2343.pdf},
  urlALT =     {http://aclanthology.coli.uni-saarland.de/pdf/W/W17/W17-2343.pdf},
  doi       = {10.18653/v1/W17-2343},
  month     = AUG,
  abstract  = {We present the work-in-progress of automating the classification of
    doctor-patient questions in the context of a simulated consultation with a
    virtual patient. We classify questions according to the computational strategy
    (rule-based or other) needed for looking up data in the clinical record. We
    compare "traditional" machine learning methods (Gaussian and Multinomial
    Naive Bayes, and Support Vector Machines) and a neural network classifier
    (FastText). We obtained the best results with the SVM using semantic
    annotations, whereas the neural classifier achieved promising results without
    it.},
  } 
  ```

   Note that these data were manually collected from books aimed at medical consultation and clinical examination, as well as resources for medical translation. These sources also need to be referenced as follows: 

   * Barbara Bates and Lynn S Bickley. 2014. 
   *Guide de l’examen clinique-Nouvelle édition 2014.*  
   Arnette- John Libbey Eurotext.
   
   * Claire Coudé, Franois-Xavier Coudé, and Kai Kassmann. 2011. 
   *Guide de conversation médicale - français-anglais-allemand.* 
   Lavoisier, Médecine Sciences Publications.

   * Owen Epstein, David Perkin, John Cookson, and David P. de Bono. 2015. 
   *Guide pratique de l’examen clinique.*  
   Elsevier Masson, Paris.
   
   * Pastore, 2015 Félicie Pastore. 2015. 
   *How can I help you today? Guide de la consultation médicale et paramédicale en anglais*. 
   Ellipses, Paris.

   * [UMVF/Medical English Portal](http://anglaismedical.u-bourgogne.fr/)  
   UFR Médecine de Dijon (Last access: May 2017)
