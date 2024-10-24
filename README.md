# Project Report: 240919~
## 1. Inference

You can run the inference code for each model from the following notebook:

- **Notebook Path:** `./inference_code/inference.ipynb`

This notebook is configured to support running inference for comparison models. 

## 2. Results

You can see the results of the inference for each model from the following notebook:

- **Notebook Path:** `./inference_outputs/{model_name}`


# Project Report: 240916

## 1. Prompts List

The following prompts were used to generate videos with different styles using various models:

1. **Realistic Style:**  
   *"A video of a duckling wearing a medieval soldier helmet and riding a skateboard."*

2. **Cartoon Style:**  
   *"A video of a duckling wearing a medieval soldier helmet and riding a skateboard in cartoon style."*

3. **Watercolor Style:**  
   *"A video of a duckling wearing a medieval soldier helmet and riding a skateboard in watercolor style."*

## 2. Comparison Models

The video generation was performed using the following models:

- **AnimateDiff**  
- **Free-Bloom**
- **CogVideo (available on Hugging Face)**  

## 3. Inference Outputs (Results)

The results for each model are saved in the respective directories as described below:

- **AnimateDiff:**  
  Results are stored in:  
  `./inference_outputs/AnimateDiff/{pre-trained_type}/`

- **Free-Bloom:**  
  Results are stored in:  
  `./inference_outputs/Free-Bloom/`

- **CogVideo (Hugging Face):**  
  Results are stored in:  
  `./inference_outputs/CogVideoX-2b/`
