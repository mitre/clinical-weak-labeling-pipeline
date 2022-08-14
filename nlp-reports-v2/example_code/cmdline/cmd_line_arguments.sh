# #!/bin/bash

# nb: python3.6 or higher must be installed

pip install {path/to/dist/conceptextract-0.0.1.tar.gz}
pip install {path/to/en_core_sci_scibert-0.4.0.tar.gz}


#may need to chmod .py files


#alter this script for your extractions then run:
python3 make_configs_file_step0.py {YOURCONFIG.yaml}

#this creates the configuration used in the following 2 steps as a file called
#YOURCONFIG.yaml saved in your current working directory

#run the nlp pipeline only once
python3 run_nlp_pipeline_step1.py {PATH/TO/YOURCONFIG.yaml}

#run this to run the snorkel pipeline on a given ruleset with concept num specified by your config
#example RULESET = Ischemia CONCEPTNUM = 2
python3 run_snorkel_pipeline_step2.py {PATH/TO/YOURCONFIG.yaml} {RULESET} {CONCEPTNUM}

