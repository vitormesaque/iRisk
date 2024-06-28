# iRisk: A Scalable Microservice for Classifying Issue Risks Based on Crowdsourced App Reviews

## Overview

Analyzing mobile app reviews is essential for identifying trends and issue patterns that affect user experience and app reputation in app stores. A risk matrix offers a straightforward, intuitive method to prioritize software maintenance actions to mitigate negative ratings. However, manually constructing a risk matrix is time-consuming, and stakeholders need help understanding the context of risks due to varied descriptions and the volume of reviews. Therefore, machine learning-based methods are needed to extract risks and classify their priority effectively. While existing studies have automated risk matrix generation in software development, they have not explored app reviews or utilized Large Language Models (LLMs) in a scalable architecture. To address this gap, we present iRisk (scalable microservice for classifying issue Risks), a tool for generating a risk matrix based on crowdsourced app reviews using Large Language Models, specifically the LLaMA 3 (Large Language Model Meta AI). Our contributions include fine-tuning the recent LLaMA 3 to enable accurate and automated review analysis. The tool, offered as a microservice, helps manage app issues and risks, providing an automated dashboard and visualizations for decision-making, monitoring, and risk mitigation. The tool is available on Github, and a presentation about the tool can be found in this [video](https://www.youtube.com/watch?v=-NnoMxrvejk).

## Index Terms
- Opinion Mining
- Large Language Model
- App Reviews
- Risk Matrix
- Issue Prioritization

## Authors

- **Vitor Mesaque Alves de Lima**  
  Três Lagoas Campus (CPTL) Federal University of Mato Grosso do Sul (UFMS), Três Lagoas, Brazil  
  [vitor.lima@ufms.br](mailto:vitor.lima@ufms.br)

- **Jacson Rodrigues Barbosa**  
  Institute of Informatics (INF) Federal University of Goiás (UFG), Goiânia, Brazil  
  [jacson@inf.ufg.br](mailto:jacson@inf.ufg.br)

- **Ricardo Marcodes Marcacini**  
  Institute of Mathematics and Computer Sciences (ICMC) University of São Paulo (USP), São Carlos, Brazil  
  [ricardo.marcacini@usp.br](mailto:ricardo.marcacini@usp.br)

## Introduction

iRisk is a scalable microservice designed to classify issue risks based on crowdsourced app reviews. By leveraging the power of Large Language Models (LLMs), specifically the fine-tuned LLaMA 3 model, iRisk can effectively analyze app reviews, identify potential issues, and prioritize them using a risk matrix. This tool aims to facilitate better decision-making, monitoring, and risk mitigation for app developers and stakeholders.

## Features

- **Automated Review Analysis**: Uses the fine-tuned LLaMA 3 model to analyze app reviews and extract relevant issues.
- **Risk Matrix Generation**: Automatically generates a risk matrix to help prioritize maintenance actions.
- **Scalable Architecture**: Implemented as a microservice, allowing multiple containers to run concurrently for efficient processing.
- **Dashboard and Visualizations**: Provides an automated dashboard with visualizations for easy decision-making.

## Getting Started

### Prerequisites

- Docker
- OLLAMA

###Ollama
```bash
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

1. Configure the repository

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
 ```
2. Install the NVIDIA Container Toolkit packages
```bash
   sudo apt-get install -y nvidia-container-toolkit
```
3. Configure Docker to use Nvidia driver
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

4. Start the container
```bash
 docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/irisk.git
    cd irisk
    ```

2. Create an OLLAMA container with the fine-tuned LLaMA 3 model:
    ```bash
    ollama create irisk -f ./Modelfile
    ```

### Usage

To run the iRisk microservice, execute the following command:
```bash
ollama run irisk


### Example Output

```bash
{
    "issue": [
        "App frequently crashes when booking a ride.",
        "Estimated arrival time calculation is rarely accurate.",
        "App can't find available drivers in some areas."
    ],
    "feature": [
        "App stability",
        "Driver arrival time",
        "Driver allocation"
    ],
    "sentence": [
        "The app frequently crashes when I'm trying to book a ride, causing me to miss important trips.",
        "Additionally, the estimated arrival time calculation is rarely accurate, which is frustrating when I'm in a hurry.",
        "Another problem I've encountered is that in some parts of the city, the app simply can't find any available drivers, even though I see several cars from the service on the street."
    ],
    "severity": [4, 3, 2],
    "likelihood": [4, 3, 3]
}
```

###Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

###License

This project is licensed under the MIT License. See the LICENSE file for details.



