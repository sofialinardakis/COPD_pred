# COPD_pred
predicting model &amp; visualization for COPD prediction based on dataset for mortalities
dataset: [https://www.kaggle.com/datasets/saurabhshahane/in-hospital-mortality-prediction](url)

**COPD DATA ANALYSIS AND MODELLING PROJECT**

DISCLAIMER: CF != COPD, initially wanted CF database (non existent for now), COPD available from mortalities database
Makes sense since 300 million people have copd…

**Hypothesis//expectations:**

- SP O2: reduced
- Respiratory rate: higher
- Glucose levels: higher
- Depression
- Diabetes 
- Decreased exercise
- deficiency anemias
- weight: lower
- Resp infections


**CONCLUSIONSSSSS**

**IN GENERAL, COPD patients tend to (according to dataset pair-plot…):**
- (High) glucose levels 
    - Between 120-163 female & male, mean=144

      (65: patients >= 130: level, 55 >= 140, 40 >= 150, 31 >= 160 … 14 >= 200, 10 >= 210)
      (Normal: 70-139)
      
- (Low) SP O2 %
    - Between 95-97 male & 93-96 female
    
    (33 <= 95%, 17 <= 94%, 12 <= 93%, 8 had <= 92%)
    (normal in copd patients, Lower than normal: 95-100)

- (High) Respiratory rate 
    - Between 17.5-22 male & 21-25 female
    
    (81 >= 17, 54 >= 20, 42 >= 22, 10 >= 25, 5 >= 27)
    (Normal: 12-20)

- Age between 64-82 male & 75-90 female
- 10/89 had depression
- 26/89 had deficiency anemias
- 26/89 had diabetes
- 7/89 had outcome death


**INFO & STATS (from the internet)**
- More than 75 percent of people with CF are diagnosed by age 2. More than half of the CF population is age 18 or older.
- Pulmonary fibrosis is lung scarring that usually occurs in older age from unknown or environmental causes. Cystic fibrosis is a genetic condition that a person is born with that causes thickened mucus in the lungs, intestines, pancreas, kidneys, and liver.
- 300 million copd, 105,000 cf
- COPD caused by exposure to environmental irritants (tobacco…)
- COPD diagnosed in adulthood
- CF is genetic, gene mutations
- CF diagnosed in childhood


**OTHER**

- while researching I came across the RNA symbols for cystic fribrosis:
		HMOX1, EDNRA, GSTM3
	and seeing AlphaFold’s predicted 3d structure (via genecards) for these genes was inspiring and motivates me to go further into what ml has to offer the world
		- dna methylation (expression), 
- HMOX1 and GSTM3 in nasal epithelial samples; 
- HMOX1 and EDNRA in blood samples 
- (useful for future projects with samples and probable diseases)
- DNA methylation at modifier genes of lung disease severity is altered in cystic fibrosis









**MODEL RESULTS conclusion** -->
**Best**; GaussianProcessClassifier, LogisticRegression, SVC







**MODEL RESULTS**

GaussianNB (WORST)
Matrix1: 
 [[150 182]
 [  6  16]]
Accuracy1:  46.89265536723164

GaussianProcessClassifier (best)
Matrix2: 
 [[332   0]
 [ 22   0]]
Accuracy2:  93.78531073446328

RandomForestClassifier (2 best)
Matrix3: 
 [[332   0]
 [ 22   0]]
Accuracy3:  93.78531073446328

LogisticRegression (best)
Matrix4: 
 [[332   0]
 [ 22   0]]
Accuracy4:  93.78531073446328

SVC (best)
Matrix5: 
 [[332   0]
 [ 22   0]]
Accuracy5:  93.78531073446328

KNeighborsClassifier
Matrix6: 
 [[331   1]
 [ 22   0]]
Accuracy6:  93.50282485875707

MLPClassifier
Matrix7: 
 [[327   5]
 [ 22   0]]
Accuracy7:  92.37288135593221
