#SignVision

## 📊 Project Results & Performance

After training for 10 Epochs, the model achieved a **Validation Accuracy of 97.93%**.

### Training Metrics

Epoch 10 Loss: 192.30
Accuracy: 97.93%

### Confusion Matrix

The confusion matrix below visualizes the performance of the classification algorithm. As seen in the diagonal density, the model predicts the vast majority of classes correctly.
Based on the screenshot of your repository structure, here is exactly how to add those images to your `README.md`.


![Confusion Matrix](Project/Images/Screenshot%202025-12-04%20at%2018.56.42.png)


**For the Activation Maps:**


![Activation Maps](Project/Images/Screenshot%202025-12-04%20at%2018.54.43.png)


**For the Grad-CAM:**


![Grad-CAM Visualization](Project/Images/Screenshot%202025-12-04%20at%2018.55.11.png)


**For the Repo Structure (The screenshot you just sent):**


![Repo Structure](Project/Images/Screenshot%202025-12-04%20at%2019.06.13.png)


### Classification Report

Detailed precision, recall, and F1-scores for all classes:

              precision    recall  f1-score   support

           A       1.00      1.00      1.00         1
           B       1.00      1.00      1.00         1
           C       1.00      1.00      1.00         1
           D       1.00      1.00      1.00         1
           ...     ...       ...       ...        ...
           X       1.00      1.00      1.00         1
           Y       1.00      1.00      1.00         1
           Z       1.00      1.00      1.00         1
         del       0.00      0.00      0.00         0
     nothing       1.00      1.00      1.00         1
       space       1.00      1.00      1.00         1

    accuracy                           1.00        28
   macro avg       0.97      0.97      0.97        28
weighted avg       1.00      1.00      1.00        28

-----

## 🧠 Model Interpretability

To understand how the CNN makes decisions, we utilized visualization techniques to peek inside the "black box."

### Activation Maps

The image below shows the output of the second convolutional layer (`conv2`). These maps highlight the specific features (edges, curves, textures) the network is detecting at this stage.

### Grad-CAM (Class Activation Mapping)

We used Grad-CAM to visualize where the model "looks" when making a prediction. The heatmap overlays the original image, showing the regions of the hand that contributed most to the classification decision.

-----

## SYLLABUS & RESOURCES

Below is the curriculum followed to build this project, ranging from Python basics to Model Interpretation.

## WEEK 0: Installation of Python and Anaconda

### Python

  - [Getting Started with Python in VS Code (Official Video)](https://www.youtube.com/watch?v=D2cwvpJSBX4&t=104s)
  - [Getting Started with Python in VS Code](https://code.visualstudio.com/docs/python/python-tutorial)

### Anaconda and Jupyter Notebook

  - [How to Install Anaconda and Jupyter Notebook on Windows 11](https://www.youtube.com/watch?v=WOK9HeB-OmY)
  - [Install Anaconda Python, Jupyter Notebook And Spyder on Mac](https://www.youtube.com/watch?v=drbaFALFKDg)
  - [Install Anaconda Python, Jupyter Notebook, Spyder on Ubuntu 22.04 LTS Linux](https://www.youtube.com/watch?v=7-naqq9fvZE)

-----

## WEEK 1–2: Python, NumPy, Pandas, Matplotlib, Git, GitHub

### Python

**Video-Based** - [Python Tutorial](https://www.youtube.com/watch?v=VchuKL44s6E&t=199s)

**Text-Based** - [Python Tutorial | W3Schools](https://www.w3schools.com/python/)

**Hindi Resources** - [Python Tutorial for Beginners](https://www.youtube.com/watch?v=vLqTf2b6GZw)

**Official Documentation**

  - [Python Documentation](https://docs.python.org/3/)

### NumPy

**Video-Based** - [Python NumPy Tutorial for Beginners](https://www.youtube.com/watch?v=QUT1VHiLmmI&t=2s)

**Text-Based** - [NumPy Tutorial (GitHub)](https://github.com/KeithGalli/NumPy/blob/master/NumPy%20Tutorial.ipynb)

  - [NumPy Tutorial | W3Schools](https://www.w3schools.com/python/numpy/default.asp)

**Hindi Resources** - [Numpy Tutorial in Hindi](https://www.youtube.com/watch?v=Rbh1rieb3zc)

**Official Documentation**

  - [Numpy Documentation](https://numpy.org/doc/stable/user/index.html#user)

### Pandas

**Video-Based** - [Complete Python Pandas Data Science Tutorial](https://www.youtube.com/watch?v=vmEHCJofslg)

**Text-Based** - [Pandas Data Science Tutorial (GitHub)](https://github.com/KeithGalli/pandas/blob/master/Pandas%20Data%20Science%20Tutorial.ipynb)

  - [Pandas Tutorial | W3Schools](https://www.w3schools.com/python/pandas/default.asp)

**Hindi Resources** - [Python Pandas Tutorial in Hindi](https://www.youtube.com/watch?v=RhEjmHeDNoA&t=1893s)

**Official Documentation**

  - [Pandas Documentation](https://pandas.pydata.org/docs/)

### Matplotlib

**Video-Based** - [Matplotlib Crash Course](https://www.youtube.com/watch?v=3Xc3CA655Y4&t=1s)

**Text-Based** - [Matplotlib Cheatsheets — Visualization with Python](https://matplotlib.org/cheatsheets/)

  - [Matplotlib Tutorial | W3Schools](https://www.w3schools.com/python/matplotlib_intro.asp)

**Hindi Resources** - [Python Matplotlib Tutorial in Hindi](https://www.youtube.com/watch?v=vBCXsAd_swk)

**Official Documentation**

  - [Matplotlib Documentation](https://matplotlib.org/stable/users/index.html)

### Git and GitHub

**Video-Based** - [Git and GitHub for Beginners - Crash Course](https://www.youtube.com/watch?v=RGOj5yH7evk&t=1900s)

**Text-Based** - [Git Tutorial](https://www.w3schools.com/git/default.asp)

**Hindi Resources** - [Complete Git and GitHub Tutorial for Beginners](https://www.youtube.com/watch?v=Ez8F0nW6S-w)

**Official Documentation**

  - [Git and Github Documentation](https://git-scm.com/docs)

-----

#### 📌 Assignment 1 (End of Week 2)

-----

## WEEK 3: Linear and Logistic Regression

### Linear Regression

**Video-Based** - [Linear Regression Algorithm | Edureka](https://www.youtube.com/watch?v=E5RjzSK0fvY)

**Text-Based** - [Python Machine Learning – Linear Regression](https://www.w3schools.com/python/python_ml_linear_regression.asp)

**Hindi Resources** - [Linear Regression Implementation | Machine Learning in Hindi](https://www.youtube.com/watch?v=e5owujIppJY)

  - [Linear Regression with Examples & Calculations](https://www.youtube.com/watch?v=zUQr6HAAKp4)

### Logistic Regression

**Video-Based** - [Logistic Regression in Python | Edureka](https://www.youtube.com/watch?v=VCJdg7YBbAQ)

**Text-Based** - [Python Machine Learning – Logistic Regression](https://www.w3schools.com/python/python_ml_logistic_regression.asp)

**Hindi Resources** - [Logistic Regression with Example](https://www.youtube.com/watch?v=r8OjlgWpAI0)

-----

#### 📌 Assignment 2 (End of Week 3)

-----

## WEEK 4-5: Neural Networks and Convolutional Neural Networks

**Video-Based** - [Deep Learning Crash Course for Beginners](https://www.youtube.com/watch?v=VyWAvY2CF9c&t=325s)

  - [Wix Studio](https://www.youtube.com/watch?v=kY14KfZQ1TI&list=PLCC34OHNcOtpcgR9LEYSdi9r7XIbpkpK1)

**Text-Based** - [CNN Explainer](https://poloclub.github.io/cnn-explainer/)

  - [PyTorch Tutorial](https://www.tutorialspoint.com/pytorch/index.htm)

**Hindi Resources** - [What is Convolutional Neural Network (CNN)](https://www.youtube.com/watch?v=hDVFXf74P-U)

  - [PyTorch for Beginners](https://www.youtube.com/watch?v=QZsguRbcOBM&list=PLKnIA16_Rmvboy8bmDCjwNHgTaYH2puK7)

-----

#### 📌 Assignment 3  (End of Week 4)

-----

#### 📌 Assignment 4  (End of Week 5)

-----

## WEEK 6: Model Interpretation — Grad-CAM

**Video-Based** - [Visualizing CAMs](https://www.youtube.com/watch?v=krtM8TuBGeA)

  - [Grad-CAM with Python](https://www.youtube.com/watch?v=9NtEMwzPDZ4)

**Text-Based** - [Grad-CAM for Explaining Computer Vision Models](https://adataodyssey.com/grad-cam/)

  - [Grad-CAM In PyTorch](https://medium.com/@codetrade/grad-cam-in-pytorch-a-powerful-tool-for-visualize-explanations-from-deep-networks-bdc7caf0b282)
  - [Grad-CAM: Visual Explanations from Deep Networks](https://glassboxmedicine.com/2020/05/29/grad-cam-visual-explanations-from-deep-networks/)

-----

## WEEK 7–8: Project

**Repo Structure:**

-----

## Acknowledgements

Parts of this project were adapted from [WINTER-PROJECT-ACTIVE-LEARNING](https://github.com/Bhavishya-Gupta/WINTER-PROJECT-ACTIVE-LEARNING) by Bhavishya Gupta.

```
```
