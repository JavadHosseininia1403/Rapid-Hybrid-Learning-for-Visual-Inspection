# Rapid-Hybrid-Learning-for-Visual-Inspection (RHL)
# 1- Introduction
Most of the natural material inspection systems suffer by frequently changes in visual appearance between batches of production. Moreover, there are some visual sorting problems that collecting and labeling numerous training samples require destructive tests. Thous problems are including wood strength estimation, metal ductility fracture prediction, prediction of structural failure (e.g. prediction of wind turbine emergency stop) and etc. Hence, training with a huge amount of labeled samples is not feasible from time and cost efficiency point of view. 

The appearance variations of natural materials such as building stone tiles, wooden boards, and medical samples against the original training materials tend to decrease the classification performance. Repeatedly collecting and labeling complete representative sample sets is a tedious effort that usually limits the accuracy and consequently the efficiency of the visual inspection system. 

Since labeling any training samples is an error-prone process and limits the performance of both supervised and semi-supervised learning schemes, categorizing training samples should be done after training feature representation phase. Therefore, the objective of devising RHL is category representation and dimension reduction in an unsupervised manner and diminishing the number of samples that need human judgment in a supervised training phase.

This post is an explanation of the developed RHL which has been successfully applied in different visual inspection problems like pencil slat and medical fundus images classification. Detailed description and results can be found in the original paper (Rapid Hybrid Learning for Visual Inspection, the link will provide soon). Also, the pencil slat data set which consist of 10 000 ground truth labeled samples is available in this post. 

# 2- How the RHL performs
We implemented the RHL in two major phase. First; an off-line unsupervised approach to build a powerful representation and dimension reduction tool. Second; an on-line supervised classifier training approach by employing active learning with uncertainty sampling. The proposed approach enables classifier to learn the category distribution space incrementally with new critical samples (which are close to the category boundaries), while it is performing the classification process on stream of data.

The developed algorithm of RHL consist of below steps.

begin{itemize}

    \item Loading and normalizing data
    
    \item Representation learning using Convolutional Auto-Encoder (CAE)
    
    \item Demonstration and evaluation of representation learning using human vision and UMAP
    
    \item Initial training of the random forest classifier
    
    \item Uncertainty sampling, and classifying samples (with no uncertainty in category) in data stream 
    
    \item Performance evaluation 
    
\end{itemize}


We will discuss each step of RHL and present Python cods in detail in later sections

