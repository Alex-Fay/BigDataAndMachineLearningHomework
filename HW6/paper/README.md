#A Summary of Brain MR Segmentation through Fuzzy Expextation Maximization and K means

## Github: https://github.com/anindox8/Atlas-Based-3D-Brain-Segmentation-in-T1-MRI

## Summary:
3D Brain MRI segmentation is a process in which MRIs break up the brain into small imaged slices. 
This allows for researchers to find abnormalalities in the brain by finding abnormalities in the spatial data of the brain. 
Thus, researchers have been using ML, specifically Expectation Maximization) to make this processs easier and faster. 
Manually sorting the data is tedious and often causes error between changing people.
This paper analyzes how to improve upon existing MRI algorithms to sort this data using a hidden Markov
random field model and applying it to the existing EM.

![image](proof1.png)
![image](proof2.png)

From thier results, they were able to smooth out the time stamps between each image slice and resulted
in a faster convergence algorithm compared to existing methods. The proposed FEM-KMeans (the compbined algorithm)
performed better than the classic EM.

