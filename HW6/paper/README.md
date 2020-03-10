## A Summary of Brain MR Segmentation through Fuzzy Expextation Maximization and K means

## Github: https://github.com/anindox8/Atlas-Based-3D-Brain-Segmentation-in-T1-MRI

## Summary:
3D Brain MRI segmentation is a process in which MRIs break up the brain into small imaged slices. 
This allows for researchers to find abnormalalities in the brain by finding abnormalities in the spatial data of the brain. 
Thus, researchers have been using ML, specifically Expectation Maximization) to make this processs easier and faster. 
Manually sorting the data is tedious and often causes error between changing people.
This paper analyzes how to improve upon existing MRI algorithms to sort this data using a hidden Markov
random field model and applying it to the existing EM.

![Part 1 of Proof](https://github.com/Alex-Fay/BigDataAndMachineLearningHomework/blob/master/HW6/paper/proof1.PNG)
![Part 2 of Proof](https://github.com/Alex-Fay/BigDataAndMachineLearningHomework/blob/master/HW6/paper/proof2.PNG)

From thier results, they were able to smooth out the time stamps between each image slice and resulted
in a faster convergence algorithm compared to existing methods. The proposed FEM-KMeans (the compbined algorithm) performed better than the classic EM. Unfortunetly, no specific numbers on time were added to the paper. 

The github provides additional details to this paper (some unmentioned parts) including the comparison between different algorithms (seen in the box and whiskers read me chart below). Additionally, graphs are provided showing the differences between tissue type which the combined EM algorithm detected using frequncy and intensity. 

![Image] (https://github.com/anindox8/Atlas-Based-3D-Brain-Segmentation-in-T1-MRI/blob/master/reports/images/res01.png)
