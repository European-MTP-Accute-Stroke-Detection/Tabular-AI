
<a name="stroke-backend"></a>
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
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#file-description">File Description</a></li>
        <li><a href="#setup-of-models-and-firebase">Setup of Models and Firebase</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project and Repository

This repository encompasses the comprehensive backend development of the "Decision Support Tool for Acute Stroke Diagnosis," undertaken as a collaborative effort by our master's team.

The backend fulfills several vital functions, including:

* Execution of AI models: It efficiently runs the sophisticated AI models designed to aid in acute stroke diagnosis.
* Generation of AI model explanations: The backend creates concise and interpretable explanations for the predictions made by the AI models, ensuring transparency and user trust.
* Database management with Google Firebase: It effectively manages data storage, retrieval, and interaction with the Google Firebase platform, promoting seamless and secure information handling.
* Additional functionalities: The backend is designed to handle various other tasks essential for the smooth functioning of the decision support tool, making it versatile and adaptive.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

You need an environment in anaconda with the correct packages to run the project on your own computer. 

### Prerequisites

To execute the code, additional python packages are required. For this, [requirements](requirements.txt) contains the package version used in this project. The installation can be done as follows:

* pip
  ```sh
  pip install -r path/to/requirements.txt
  ```

### File Description

This project includes several python scripts and folders. The following list serves as a brief explanation.

* [01_EDA](01_EDA.ipynb) - Loading the data and brief visual time series analysis. 
* [02_Preprocessing](02_Preprocessing.ipynb) - Processing of the data to split the information and prepare it for training. 
* [03_Modeling](03_Modeling.ipynb) - AI training with dataset creation.
* [04_1_Artificial_Data](04_1_Artificial_Data.ipynb) -  Creation of artificial time series for AE and XAI tests. 
* [04_Evaluation_Complete](04_Evaluation_Complete.ipynb) - Evaluation of AE results for complete time series.
* [04_Evaluation_Segment](04_Evaluation_Segment.ipynb) - Evaluation of AE results for segmented time series.
* [05_Explainability](05_Explainability.ipynb) - Application of XAI techniques covered in this thesis.
* [PCE_AI](PCE_AI.ipynb) - AI pipeline for deployment testing. Contains the final preprocessing used for the thesis submission.


### Setup of Models and Firebase

For the successful execution of the backend, it is imperative to ensure two key steps: first, the correct linkage of models to the main code, and second, the accurate configuration of Firebase access tokens in the database functionalities. 


<!-- USAGE EXAMPLES -->
## Usage


Once all components on the Python side are meticulously set up, and the trained models are seamlessly integrated with the code and data access to Firebase is established, the backend becomes fully prepared for execution.

```sh
python main.py 
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- LICENSE -->
## License

Distributed under the MIT License.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Patrick Knab: [Github Profile](https://github.com/P4ddyki)

Marc Becker: [Github Profile](https://github.com/beckmarc)

Project Link: [https://github.com/European-MTP-Accute-Stroke-Detection/stroke-backend](https://github.com/European-MTP-Accute-Stroke-Detection/stroke-backend)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Helpful Resources:

* [LIME](https://github.com/marcotcr/lime)
* [SHAP](https://github.com/shap/shap)
* [Grad-CAM](https://keras.io/examples/vision/grad_cam/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
