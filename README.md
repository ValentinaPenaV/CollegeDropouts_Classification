# Classification Analysis to predict College Dropouts from OpenUniversity data
I used a **classification** approach to build a predictive model for student- and course-level dropouts using the Open University publicly available dataset. I first focused on **thorough Exploratory Data Analysis (EDA)** to understand and merge the data at hand before building a **Gradient Boosted Tree Classification Model (XGBoost)** to predict whether students would drop out of a specific class they were taking or not.

## Project Intro/Objective
Many reputable sources such as the [Washington Post](https://www.washingtonpost.com/news/grade-point/wp/2018/06/08/why-do-so-many-students-drop-out-of-college-and-what-can-be-done-about-it/), [The New York Times](https://www.nytimes.com/interactive/2019/05/23/opinion/sunday/college-graduation-rates-ranking.html), and [Forbes](https://www.forbes.com/sites/prestoncooper2/2017/12/19/college-completion-rates-are-still-disappointing/#1666e197263a) have in recent years described a "crisis" with regards to college dropout rates. According to these articles as little as 57% of students finish their undergraduate degrees, with completion rates dropping to 47% for community colleges. These numbers are alarming not only for the state of education within the United States; this issue also apparently has an important impact on the country's economy. Indeed, not having an undergraduate degree in this day and age can be a hurdle for qualifying for high-level jobs, which in turn often mean lower wages, more difficulty to repay debt accrued through student loans, and less opportunities for social mobility overall. 
Understanding factors that increase a student's propensity to drop out is, in my opinion, a worthwhile endeavor and a potentially important first step in tackling this crisis. This analysis therefore focuses on identifying which attributes - both course- and demographic-related - are useful in predicting drop-outs. By understanding them we can work towards bulding a profile of at-risk students and reform courseload and structure to address the issue at hand.

## Datasets Used
* Open University Learning Analytics dataset from [Kaggle](https://www.kaggle.com/rocki37/open-university-learning-analytics-dataset)

## Methods Used
* Exploratory Data Analysis (EDA)
* Data Resampling (for imbalanced classes)
* Classification Modeling
* Data Visualization

## Notable Technologies Used
* Python 3, Jupyter Notebook
* IMBlearn **> Data Resampling**
* Scikit-learn,  XGBoost **> Classification Modeling**
* Pandas, Numpy, Matplotlib, Seaborn, Tableau **> Data Processing/Visualization tools**
* AWS **> Cloud Computing for processing power**
* etc.

## Main Analysis Threads
* **Data Aggregation** - Merged data of interest hosted across various tables with shared primary and secondary keys.
* **Data Cleaning** - Conducted Exploratory Data Analysis to understand patterns in missing data.
* **Feature Engineering** - Transformed data in preparation for analysis as per the patterns uncovered through EDA.
* **Classification Modeling** - Applied a Gradient Boosted Tree Classifier to predict student dropout using the features selected.

## Model 
### Data Cleaning and Feature Engineering
#### Dealing with meaningful NaNs:
* Missing values for assessment due-dates were observed for all types of final result *except* for withdrawals but were also *only* observed for exams and not other forms of assessment. Missing values were therefore filled with _-999_
![Assessment due-date by result diagnostic plot]()
![Assessment due-date by assessment diagnostic plot]()

* Missing scores were observed for all types of final result *except* for distinctions, but were *only* observed for Tutor-Marked Assessments. Score were therefore either imputed as the mean of the student's previous assessments in the module or were set to 0 if there were no prior scores avaiable.
![Missing scores by result diagnostic plot]()
![Missing scores by assessment diagnostic plot]()

#### Recoding feature values:
* "_date_registration_", the feature expressing the amount of time between module registration and course start (in days) was multiplied by -1 so that it represented how many days **before** the course a student had registered. Negative values thereby indicated late registration.
* "_disability_" was recoded as binary (0: no, 1: yes)
* "_gender_" was recoded as binary (0: male, 1: female)
* "_final_result_" was recoded as binary (0: _Pass/Fail/Distinction_, 1: _Withdrawn_)

#### Feature engineering:
* Created binary variable "_late_assessment_submission_" expressing whether a student had submitted **any** assessments late (0: no, 1: yes)
* Created binary variable "_late_registration_" expressing whether a student has registered late for the course (0: no, 1: yes)
* Age variable made ordinal (0: _0-35 years_, 1: _35-55 years_, 2: _55+ years_)
* Highest level of attained education variable made ordinal (0: _No formal qualifications_, 1: _Lower than A-levels_, 2: _A-levels or equivalent_, 3: _Higher Education_, 4: _Postgraduate Education_)

**> Any missing values not addressed above were dropped** 

### Classification Modeling
Out of multiple classification models tested with both a full and reduced set of features (Logistic Regression Classifier, kNN, Naive Bayes, Decision Tree Classifier, Random Forest Classifier, Gradient Boosted Tree Classifier), a gradient boosted tree method (**XGBoost**) yielded the best results. I measured model performance using the F1 metric, which tracks model accuracy as a function of both precision and accuracy, thereby expressing a tradeoff between the two (_see mathematical definition below_).

![F1 equation](https://wikimedia.org/api/rest_v1/media/math/render/svg/1bf179c30b00db201ce1895d88fe2915d58e6bfd)

This choice of metric was motivated by the desire to grant importance to maximizing the number of dropout cases flagged, while also being mindful of the potential costliness of indiscriminately applying interventions to individuals not at risk. Ultimately, the features included included in this model were (in order of importance): 
![XGBoost feature importances]()

### Model Visualization
In order to more easily compare and understand the attributes of each model tested, I built a custom Tableau Dashboard (_linked below_) that plots a modified confusion matrix for each and allows for interactive exploration of the individual models' validation scores (Accuracy, Recall, Precision, F1) and feature importances. 
[Tableau Dashboard Link]()


## Take-Homes
In this project I identified which features relating to both students and courses were important for predicting whether students were likely to drop out of a module or not. The factors which impacted risk of withdrawal most prominently were student disability, a lack of prior studied credits (i.e. little previous educational experience within the Open University), prior level of educational attainment, and course length. These insights are useful as they clearly point out two actionable paths to address dropout rates. On one hand, it defines an at-risk population typified by students with disabilities and low-levels of both educational attainment at large and prior educational engagement for whom targetted interventions can be designed. On the other, it suggests potential changes to the educational curriculum or structure to address factors such as overly long courses which are also associated with high student dropout rates.

## Final Notes
In the spirit of constant improvement, I believe this model would benefit from including more granular data on the quantity and quality of course material, and of the frequency of each student's interaction with it (all available within the Kaggle dataset). I may get around to working it into the analysis above.
Finally, the code here is still a work in progress - functions and code structure will be updated periodically as I work through fixes!
