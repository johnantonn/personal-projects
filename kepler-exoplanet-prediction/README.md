# kepler-exoplanet-prediction
Exoplanet candidate prediction using the KOI dataset.

![image](https://user-images.githubusercontent.com/8168416/157868177-7ed73187-c672-4dbe-a7be-c7111143b29e.png)

## Description
The Kepler Space Observatory is a NASA-build satellite that was launched in 2009 in the context of the Kepler mission. The mission's target was to search for exoplanets in external star systems and ended in 2018. Kepler was able to find planets by looking for small dips in the brightness of a star when a planet transits in front of it.

*Note*: A transit occurs when a planet passes between a star and its observer. A planetary transit is described by parameters such as the period of recurrence, the duration of the transit and the fractional change in brightness of the star.

## Dataset
This dataset is a cumulative record of nearly 10k observed Kepler objects of interest, downloaded on 24th February 2022. An extensive data dictionary can be accessed [here](https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html).

## Solution
The goal of this notebook is to classify observed Kepler objects of interest as Planet Candidates or False Positives (binary classification). The notebook is comprised of the below distinct sections:
- Exploratory Data Analysis: where insight on the dataset and its features is provided.
- Pre-processing: where things like null values, feature selection and train/test split are handled.
- Modelling: where classification algorithms are applied and evaluated in terms of performance.

## References
- https://www.nasa.gov/mission_pages/kepler/overview/index.html
- https://www.kaggle.com/nasa/kepler-exoplanet-search-results
