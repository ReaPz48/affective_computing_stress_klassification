<h1>Affective Computing: Stress Classification During Public Performances Throught The ECG-Signal</h1><br/>
This Repository holds an affective computing implementation to classify stress during public performances. The classification was made through the ECG-Signal.<br/><br/> 
<br/>
A ffective Computing is an interdisciplinary field of research that spans the fields of computer science and psychology. The idea describes a computer system that is able to recognise, interpret and simulate human emotions. Intelligent machines and devices should be able to interact with humans in a natural and empathetic way in order to support them. A ffective Computing can be realised through multiple inputs from humans. These range from facial expressions to the ECG signal of the heart.
The study in this thesis attempts to classify stress during public performances through A ffective Computing and the ECG signal. For this purpose, the ECG data of five test subjects were collected. Four of the five subjects were accompanied during a musical performance and one during a presentation. The data were pre-processed ac- cordingly so that they could be classified by a machine learning model. The models were trained and tested using the WESAD a ffective database. The performance of all trained models was evaluated through various metrics.
In total, five di erent types of models were trained: 1) Random Forest Classifier (ROC: 0.957/PRC: 0.869); 2) Support Vector Machine (ROC: 0.955/PRC: 0.896); 3) Lo- gistic Regression (ROC: 0.946/PRC: 0.874); 4) k-Nearest Neighbors (ROC: 0.944/PRC: 0.832); 5) Linear Discriminant Analysis (ROC: 0.946/PRC: 0.885). After training and testing the models using the WESAD dataset, an attempt was made to classify the self- collected data using these models. The results were as follows: RFC – ROC: 0.607/PRC: 0.709; SVM – ROC: 0.712/PRC: 0.836; LR – ROC: 0.670/PRC: 0.779; kNN - ROC: 0.600/PRC: 0.757 and LDA – ROC: 0.670/PRC: 0.844. The results of the collected data were worse, compared to the dataset before. This is mainly due to the high number of stress classifications of all musicians. For this reason, new models were only trained with the data collected from the musicians. This should show whether the result can be improved. The result was only minimally improved. Furthermore, the results show that all musicians were under much more stress compared to the presenting person. The reason for this could be the physical and mental work of making music, which is also classified as stress by the models.
Aff ective computing can be used for stress classification in the course of public performances. The unimodal approach using the ECG signal allows such a classification. The results of the classification could also be used to calculate a stress progression. This progression could serve as feedback and help people to improve.
<br/>
<br/>
<p align="center">
  <img src="./pictures/polar_brustgurt.png" width="450" title="Wireless Polar H10 ECG Belt and Sensor">
</p>

<br/>
<br/>
<p align="center">
  <img src="./pictures/ecg_plot.png" width="450" title="ECG Raw Signal from Polar H10 Sensor">
</p>

<br/>
<br/>
<p align="center">
  <img src="./pictures/C__stress_moving_average.png" width="450" title="Calculated Stress Level">
</p>
