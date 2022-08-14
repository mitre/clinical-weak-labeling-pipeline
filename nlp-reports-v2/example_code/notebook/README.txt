

Also be sure to download and install en_core_sci_scibert-0.4.0.tar.gz (or preferred clinical model) from allenai GitHub
 

To install conceptextract and language model:

>pip install path/to/dist/conceptextract-0.0.1.tar.gz
>pip install path/toen_core_sci_scibert-0.4.0.tar.gz 

0-create configs with make_configs
1-run nlp pipeline and save objects (only need to do once per dataset)
2- run snorkel pipeline
3-repeat steps 0 and 2 many times to arrive at optimal permutation


#todo: to see if package works on windows