README

The purpose of this repository is to contain code and data that will be used to describe differences between a prevalent form of leukemia and their healthy counterparts. 
Our goal is to apply recently-developed machine learning methods to synthesize gene expression and protein expression data in order to identify functional drivers of cancer. 

In more detail, we aim to identify modular functional units in Chronic Lymphocytic Leukemia (CLL) that are not present in normal B cells. 
We will accomplish this through the use of analytical methods recently invented/developed out of Worchester Polytechnic Institute by two research groups. 
First Chong Zhou and Randy Paffenroth created the robust deep autoencoder method and demonstrated it's effectiveness at denoising MNIST data without clean training data. 
Then Hongshu Cui and Dimitry Korkin applied this method on gene expression data and demonstrated this methods superior effectiveness compared to traditional clustering approaches. 

We aim to apply these methods to CLL gene expression data to better understand the differences in functional modules between healthy and cancerous cells. 
Understanding functional modules in cancer is important because it allows one to 
It is important to 

Data Inputs:
1) Leukemia_GSE22529_U133B.csv: Gene expression microarray data from the Curated Microarray Database (CuMiDa) https://sbcb.inf.ufrgs.br/cumida.
2) HuRI.psi: Protein-Protein Interaction (PPI) data from the Human Reference Interactome (HuRI). Note: unable to add due to file size but can be accessed here http://www.interactome-atlas.org/download.
3) Not yet defined: links from gene probes to gene names. 

Process:
1) Process gene expression data with robust deep autoencoders in order to filter out anomalies (noise and outliers) and generate low-dimensional embeddings of the data. 
2) Use the combination of PPI data and gene expression embeddings to perform community detection on the PPI data using the Louvain algorithm. 
   Here the role of community detection is to identify groups of proteins that are involved in performing/regulating the same process.
   This process could be done solely with the PPI data, but we choose to enhance the PPI data with our gene expression data. 
   We know that proteins that interact together are likely to be involved in the same process(es).
   We also know that proteins are the products of genes and when genes follow similar expression patterns, they are likely to share the same function. 
   So we enahnce the community detection process by labelling PPI values with their corresponding gene expression labels.
   
3) Use the community detection results as weights and perform Eisen clustering of the pre-processed gene expression data to identify clusters of similar genes, informed by PPI information. 

Outputs:
