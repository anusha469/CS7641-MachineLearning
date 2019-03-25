The code is adapted from Jonathan Tay’s Github repository. The goal is to explore unsupervised learning and dimensionality reduction. 
1. Run ‘python parse.py’ from the terminal. This creates the necessary .hdf files out of the .csv data files
2. Run 'python benchmark.py', 'python PCA.py', 'python ICA.py', 'python RP.py', 'python RF.py'  from the terminal. This will produce an output that is needed to run clustering on. 
3. Run ‘./run.bat’ to execute the clustering.py file five times with various inputs, one for the baseline dataset and once for each of the four dimensionality reduction algorithms. 
4. From the R Terminal, Run RScript Plots.R. The is are processed in plots.R and output to is written ./output/plots/