
<a name="Tabular AI"></a>
<!--

<!-- PROJECT LOGO -->
<br />

<div align="center">


  <h1 align="center">Stroke Prediction Tabular</h1>

  <p align="center">
    Decision Support Tool for Acute Stroke Diagnosis
    <br />
    <a href="https://github.com/European-MTP-Accute-Stroke-Detection"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="http://brainwatch.pages.dev">View Demo</a>
  </p>
</div>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#file-description">File Description</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project and Repository

This repository encompasses the comprehensive tabular AI development of the "Decision Support Tool for Acute Stroke Diagnosis," undertaken as a collaborative effort by our master's team.

The tabular AI fulfills several vital functions, including:

* Experimentation with different ML techniques, such as Catboost, XGBoost, GDTs, etc.
* Data preprcessing of tabular dataset containing medical stroke data.
* Experimentation with XAI for Catboost 

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

You need an environment in anaconda with the correct packages to run the project on your own computer. 

### File Description

This project includes several python scripts and folders. The following list serves as a brief explanation.

* [data](data) - Data from https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
* [Old_Scripts](Old_Scripts) - Old Scripts 
* [Scripts](Scripts) - Scripts with created objects for AI execution in backend

## Usage

The central focus of this repository is housed within the [Scripts](Scripts) directory. This directory encapsulates a collection of meticulously crafted scripts tailored to facilitate the training of diverse AI techniques, ranging from [Catboost](Scripts/catboost.ipynb) to [XGBoost](Scripts/xgboost.ipynb). Furthermore, it serves as a hub for delving into data exploration through Exploratory Data Analysis [EDA](Scripts/eda.ipynb).

Each file contains concise comments detailing its contents and the steps it encompasses. After meticulous tuning using Optuna, the CatBoost model emerged as the optimal choice, striking a harmonious balance between accuracy, F1 recall, and precision. We then went on to elucidate its workings for integration into the BrainWatch app, using SHAP to provide a comprehensive explanation.

We extended our analysis by incorporating Sascha's gradient decision tree approach ([GDT](Scripts/GDT.ipynb)) to assess its performance on our dataset. This decision was prompted by the challenges faced by other machine learning techniques in achieving satisfactory results. Despite its potential, the gradient decision tree did not surpass the performance of Catboost in our experiments. It's worth noting that this outcome could be attributed to the fact that we did not thoroughly optimize the hyperparameters of the gradient decision tree. Further investigation into hyperparameter tuning might uncover its hidden potential.

<!-- LICENSE -->
## License

Distributed under the MIT License.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Patrick Knab: [Github Profile](https://github.com/P4ddyki)

Mira Rotaru: [Github Profile](https://github.com/Mira-Rotaru)

Diana Drăgușin: [Github Profile](https://github.com/DianaDragusin)

Project Link: [https://github.com/European-MTP-Accute-Stroke-Detection/stroke-backend](https://github.com/European-MTP-Accute-Stroke-Detection/stroke-backend)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
