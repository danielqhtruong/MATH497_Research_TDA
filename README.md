# MATH497_Research_TDA
Undergraduate research paper for Data Science minor requirement with faculty mentorship. Specialize in Topological Data Analysis.

**Topological data analysis (TDA)** is a relatively new and evolving field of applied mathematics and data science that uses topological information from datasets to identify meaningful structures and features that persist across many scales. One way to illustrate some concepts from TDA is to:

1.	Consider a plot of given data points.
2.	Draw small circles of the same radius around each point.
3.	Systematically increase the radius of each circle simultaneously.

The overlapping region of each circle gives a connected component and can lead to the emergence and identification of “holes” in the dataset, i.e., when the overlapping circles form a closed loop. When a hole emerges, we say it is “born” at that radius value, referred to as its birth time. Eventually, the radii become large enough that overlapping circles cover the hole, an event called the “death” of that feature.
      
We can identify persistent, meaningful features in the dataset by tracking the births and deaths of topological features (e.g., connected components, holes, and voids). This approach has recently been applied to a broad range of important scientific questions, including studying polarization by identifying politically distinct urban areas of the US, capturing daily patterns in hurricane structures, and identifying communities with reduced access to cooling centers in extreme heat.
      
Given the devastating effects of recent wildfires in Los Angeles County, we seek to apply principles from TDA to study resource allocation and wildfire risk in Southern California. Our research questions are:

1.	Are fire prevention methods and wildfire resources equitably distributed? In other words, are there communities at high risk of wildfire with gaps in accessibility to preventative or emergency resources?
2.	Are there significant differences in access to wildfire resources for populations of different demographics?
   
To answer these questions, we propose applying TDA concepts to datasets from **CAL FIRE**, **OpenStreetMap**, and the **US Census Bureau**.



