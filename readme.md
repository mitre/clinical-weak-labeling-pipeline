
# A weak labeling NLP pipeline for metadata extraction from clinical reports 

## POCs: Chad Hunter, Emily Workman
Authors: Emily Workman, Chad Hunter, Daniel Berman, Jason Yao, Steven Guan, Laura Strickhart, Rob Beanlands, Thiery Mesana, Sybil Russell, Robert DeKemp, David Slater

Date: 2022/08/11

Purpose:
	This readme will explain how to run the NLP pipeline
for data extraction.  In addition, it will explain how to run
the prep code for data parsing (to be added later).

Rev: A
################################################################

For NLP data extraction:
There are 3 important folders: 
	lib 
	data
	NLP_DataExtract

The lib folders contains the raw NLP python code for the data
extraction process, you should not modify this unless you are 
changing the funamental way the NLP pipline works, such as adding
new detection functions etc.

The data folder contains a csv file with the clinical reports
you wish to extract data from.  The file must have the first
row as the header which you will identify to the program
which colum the important information is stored in.

The NLP_DataExtract folder has the python code used to run the
nlp, there are 3 important python scripts and a VIP csv file:
	Step1_input_arguments
	Step2_nlp_pipe
	Step3_SnorkelPipe
	VIP_FilePaths

Step1_input_arguments is the first python script you need to run, 
it will generate important input files for the NLP.

Step2_nlp_pipe is the second python sript which need to be run in
order to generate addional input data for the NLP pipeline

Step3_SnorkelPipe is the python script that will extract the data
from the clinical reports and generate data on the statistics of
the data extraction

VIP_FilePaths is a csv file that tells the scripts where the lib
folder, data folder and where you want the output folder to be
located, and where the lib data files will be stored.  Also, the
path to where you wand the snorkle evaluation stats and data
to be stored.


# Public Release
©2022 The MITRE Corporation and The Ottawa Heart Institute
Approved for Public Release; Distribution Unlimited. 
Public Release Case Number 22-2179

# License

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

`http://www.apache.org/licenses/LICENSE-2.0`

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.


