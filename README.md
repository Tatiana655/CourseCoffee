# CourseCoffee

## Project Description
“Cup Tests" (Coffee capping) is very subjective, and most often depends on the skill of the taster and his preparation, i.e. the evaluation of coffee is based on the opinion of tasters, whose experience can be very different. Moreover, not all tasters are able to find a defect in a cup of coffee. In this regard, one of the solutions to such a problem may be a recommendation system for evaluating the quality of coffee.

## Dataset
The data set contains time series recorded by the electronic E-Nose system for the application of coffee quality control focused on the detection of defects in coffee. The database includes 58 measurements of coffee, divided into three groups and designated as high quality (HQ), medium quality (AQ) and low quality (LQ). There are 8 columns for each sample (from each sensor). More than 300 points - the readings of the corresponding sensor.
Dataset: https://data.mendeley.com/datasets/7spd6fpvyk/1 (Dataset: Electronic Nose for Quality Control of Colombian Coffee through the Detection of Defects in “Cup Tests”) 

## Module architecture

![Human](https://user-images.githubusercontent.com/114859682/233766875-3c2723e6-f892-4c2c-ac70-7569a47be3f5.png)

## Metrics for tracking the operation of the system
1. Maximum probability
&nbsp
![max](https://github.com/Tatiana655/CourseCoffee/assets/114859682/0fb29591-b831-4c5b-b809-7fa406836e0e)
2. Minimum difference between probabilities
![min](https://github.com/Tatiana655/CourseCoffee/assets/114859682/107c816c-fca3-428b-8986-6896df3bb2c3)
3. Average change in maximum probabilities
![sum](https://github.com/Tatiana655/CourseCoffee/assets/114859682/cbf550bd-bd40-482f-9d21-c099294c9c7c)
