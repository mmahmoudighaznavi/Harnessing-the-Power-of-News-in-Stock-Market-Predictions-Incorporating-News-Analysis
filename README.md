# ADS599: Capstone - Harnessing News Analysis for Enhanced Stock Market Predictions



**Project Status: Active**

# Table of Contents
--------
1. [Run Locally](#Run-Locally)
2. [Project Introduction](#Project-Introduction)
3. [Project Objective](#Project-Objective)
4. [Project Overview](#project_overview)
5. [Methodology](#Methodology)
6. [Partners/Contributors](#Partners/Contributors)
7. [Methods Used](#Methods-Used)
8. [Project Framework](#Project-Framework)
9. [Modeling & Model Evaluation](#modeling_&_model_evaluation)
11. [Conclusion](#conclusion)
12. [References](#references)
13. [License](#license)
14. [Acknowledements](#acknowledgements)
--------



## Run Locally

Clone the project

```bash
  git clone https://github.com/ruddysimon/Harnessing-the-Power-of-News-in-Stock-Market-Predictions-Incorporating-News-Analysis.git
```

Go to the project directory

```bash
  cd Web-application
```

Install dependencies

```bash
  pip install pandas
  pip install numpy
  pip install yfinance
  pip install keras
  pip install streamlit
```

Start the server

```bash
  streamlit run main.py
```

## Project Introduction
The primary goal of this project is to explore the impact of news sentiment on stock market movements. We employ  deep learning models, such as GRU and LSTM, to predict stock market trends. By integrating news analysis, we aim to discover correlations between stock market behavior and the sentiment conveyed in news articles.

## Project Objective
This project endeavors to decode the intricate relationship between news sentiment and stock market fluctuations. Employing sophisticated deep learning models like GRU and LSTM, it aims to predict stock market trends and analyze the correlation with news sentiment. The research could potentially unlock new avenues in predictive analytics in finance, shedding light on how external information sources influence market dynamics.

### Methodology 

**Data Splitting in Time Series Analysis**  
We split the data into training and testing sets for time series forecasting. This split is crucial to prevent data leakage and ensure that our test data always lies in the future, relative to the training data. By doing so, we avoid the unrealistic scenario of training our model with future data, which can lead to misleading results.

**Time Series Forecasting using Overlapping Windows**  
Our approach involves utilizing overlapping windows to capture the sequential dependencies inherent in time series data. Each window encompasses a set of data points from the past to predict the next point. For example, with a window size of two, we use two consecutive data points to forecast the subsequent point. This method ensures a comprehensive understanding of the temporal patterns within the data.

**RNN and Advancements to LSTM and GRU**  
Initially, we explored Recurrent Neural Networks (RNN) for modeling stock market predictions. 
![RNN network](https://github.com/ruddysimon/Harnessing-the-Power-of-News-in-Stock-Market-Predictions-Incorporating-News-Analysis/blob/main/images/rnn.png)

However, we found that RNNs have limitations in retaining sequential information over extended periods. To overcome this, we transitioned to more advanced models like Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU). These models are better equipped to handle the complexities of stock market prediction by effectively maintaining information across longer sequences.

## Partners/Contributors

**Ruddy Simonpour:** 
Data Scientist with over 2 years of experience. Former AI/Data Analyst Intern at City of Palo Alto, Data Scientist at GetMotivatedBuddies, and currently working at the University of San Diego. Expertise in predictive modeling, transformers, machine learning, and data analysis.

Email: Ruddys@sandiego.edu  
Website: www.ruddysimon.com

**Mohammad Mahmoudighaznavi:**  
Data Scientist with expertise in machine learning and data analysis.  
Email: mmahmoudighaznavi@sandiego.edu

---
## Methods Used    
- [ ] Data Exploration
- [ ] Predictive Modeling
- [ ] Statistical Analysis
- [ ] Sentimental Analysis
- [ ] Data Modeling
- [ ] Data Pre-processing
- [ ] Visualization
- [ ] Natural Language Processing (Langchain, LLM)

## Technologies
- [ ] Python


## Project framework

<p align = "center">
  <img src="https://github.com/ruddysimon/Harnessing-the-Power-of-News-in-Stock-Market-Predictions-Incorporating-News-Analysis/blob/main/images/framework.png">
</p>
  
## License

[MIT](https://choosealicense.com/licenses/mit/)

Copyright (c) 2023 Mohammad Mahmoudighaznavi, Ruddy Simonpour

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Acknowledgments
We extend our sincere gratitude to [Dr. Ebrahim Tarshizi](https://onlinedegrees.sandiego.edu/faculty/ebrahim-k-tarshizi/), who played a pivotal role in the realization of this project as part of our ADS599 capstone course. As the Data Science and Artificial Intelligence Program Director at the University of San Diego (USD), Dr. Tarshizi’s guidance, expertise, and invaluable insights have been instrumental in shaping the direction and success of our work. His contributions to our learning journey and project development are deeply appreciated.


