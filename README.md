# AI-sha #

### 1. Overview ### 

AI-sha is a Deep Learning Chatbot is an AI-powered chatbot that uses GPT-3 to generate responses to user inputs.
The chatbot can learn from the internet and remember every sentence ever talked
The chatbot also can be fine-tuned using a dataset of text inputs and corresponding outputs. 

### 2. Installation ### 

To use the Deep Learning Chatbot, you will need to install the following dependencies:

⦁	`nltk`: A library for natural language processing tasks such as tokenization and lemmatization.
⦁	`sklearn`: A library for machine learning tasks such as feature extraction and similarity measurement.
⦁	`transformers`: A library for state-of-the-art natural language processing models such as GPT-3.
⦁	`torch`: A library for deep learning tasks such as model fine-tuning and training.

You can install these dependencies by running the following command:

```pip install nltk sklearn transformers torch```

### 3. Usage ### 

To use the chatbot, you need to create an instance of the `DeepLearningChatbot` class:

```chatbot = DeepLearningChatbot()```

#### 3.1 Initialization #### 

```pip install -r requirements.txt && python -c "from deep_learning_chatbot import DeepLearningChatbot; dlc = DeepLearningChatbot(); dlc.fine_tune_gpt3('model_name', 'dataset_path', batch_size=4, device='cuda')"```

This command will first install the necessary libraries and dependencies using pip, and then it will initialize the AI by creating an instance of the DeepLearningChatbot class and fine-tune the GPT-3 model using the specified dataset.

Where model_name is the name of the GPT-3 model you want to use, dataset_path is the path to the dataset you want to use for fine-tuning, batch_size is the batch size for fine-tuning, and device is the device to use for fine-tuning.

It's important to note that model_name should be one of the models available in the transformers library, for example : "distilgpt2" , "gpt2" or "openai-gpt" and dataset_path should be the path to your dataset where you want to fine-tune the model.

**The GPT-3 model you should use depends on your specific use case and the amount of computational resources you have available.**

 - If you want to use the smallest model with the lowest computational requirements, you can use distilgpt2.
 - If you want to use a model with a good balance of performance and computational requirements, you can use gpt2.
 - If you want to use the largest model with the highest performance, you can use openai-gpt.
 - It's also worth noting that GPT-3 models are very large and they require a lot of memory to be loaded and fine-tuned. So, it's recommended to have a powerful GPU with at least 16GB of memory.

You can also consider using a smaller version of GPT-3 like gpt2-medium or gpt2-small, which are more memory-efficient and still can perform well on most tasks.

It's also a good idea to test the different models with your specific dataset and use case, and compare their performance.

#### 3.2 Learning from the Internet ####

To learn from the internet, you can use the learn_from_internet method, which takes a string of text as a parameter:

```text = "This is an example of text that the chatbot can learn from the internet"```

#### 3.3 Fine-tuning the GPT-3 model #### 

To fine-tune the GPT-3 model, you can use the `fine_tune_gpt3` method, which takes the following parameters:

⦁	`model_name`: The name of the GPT-3 model to use (e.g., "distilgpt2").
⦁	`train_texts`: A list of texts to use for fine-tuning the model.
⦁	`train_labels`: A list of labels corresponding to the texts.
⦁	vbatch_size`: The number of samples to use in each training iteration.
⦁	`device`: The device to use for training (e.g., "cuda" for a GPU).

```
model_name = "distilgpt2"
train_texts = ["This is an example of text that the chatbot can learn from","This is an example of text that the chatbot can learn from"]
train_labels = [0,1]
batchbatch_size = 4
device = "cuda"
```

In the context of this script, the `device` parameter specifies which device the model and data should be loaded on to perform the computation.

The value 'cuda'` means that the script will use the GPU (if available) to perform the computation. CUDA (Compute Unified Device Architecture) is a parallel computing platform and API developed by NVIDIA that allows using the GPU for general purpose computing. If a GPU with CUDA support is available, it will allow the model to perform computation much faster than using a CPU.

The value 'cpu' means that the script will use the CPU to perform the computation.

It's important to note that in order to use the 'cuda' option, your system must have a NVIDIA GPU with CUDA support and you must have the CUDA toolkit and NVIDIA drivers installed on your system.

In the context of this script, the `batch_size` parameter determines the number of samples in a single batch of data that is passed through the model at once during training.

A larger batch size allows the model to make better use of the GPU and perform more computation in parallel, which can lead to faster training times. However, it also requires more memory to store the activations and gradients.

A smaller batch size allows the model to use less memory, but it will also result in slower training times.

When you fine-tune a model, you will generally want to use a batch size that is as large as possible while still fitting into the GPU memory.
However, it also depends on the specific dataset and the size of the model, so you may need to experiment with different batch sizes to find the best value for your use case.

It's also worth noting that in general a batch size of 4 is a good starting point for fine-tuning GPT-3 models, but you may need to increase or decrease it depending on the specific dataset and the GPU memory available.

#### 3.4 Chatting ####

To start chatting with the chatbot, you can use the `chat` method:

```chatbot.chat()```

This will start a loop where the user can input a sentence, and the chatbot will generate a response. The loop will continue until the user inputs "bye".

#### 3.5 Saving and loading the fine-tuned model ####

The chatbot allows you to save and load the fine-tuned model, optimizer, and scheduler state, respectively. You can use the `save_model` method which takes the path where the model should be saved, and `load_model` takes the path where the model is saved.

```path = "path/to/save/model"```

and to load the model:

```path = "path/to/save/model"```

### 4. Troubleshooting ### 

⦁	If the chatbot is not generating good responses, it might be because the fine-tuning process did not converge, or the dataset was not adequate.
⦁	If the chatbot is not learning from the internet, it might be because the text passed to the `learn_from_internet` method is not adequate.
⦁	If the code is not running correctly, it might be because the dependencies are not correctly installed, or the versions are not compatible.

### 5. Conclusion ###

The Deep Learning Chatbot is an AI-powered chatbot that uses GPT-3 to generate responses to user inputs. The chatbot can be fine-tuned using a dataset of text inputs and corresponding outputs. The chatbot can also learn from the internet and remember every sentence ever talked. 

### 6. Legal Disclaimer:
The information provided on this GitHub page (the "Page") is for general informational purposes only. The Page is not intended to provide legal advice or create an attorney-client relationship. You should not act or rely on any information on the Page without seeking the advice of a qualified attorney. The developer(s) of this Page do not warrant or guarantee the accuracy, completeness, or usefulness of any information contained on the Page and will not be liable for any errors or omissions in the information provided or for any actions taken in reliance on the information provided.

### 6.1 Policy: ### 
All code and other materials provided on this Page are the property of the developer(s) and are protected by copyright and other intellectual property laws. You may not use, reproduce, distribute, or create derivative works from the code or materials on the Page without the express written consent of the developer(s). If you would like to use any of the code or materials provided on this Page for any purpose, please contact the developer(s) for permission.

The developer(s) reserve the right to make changes to the Page and to the code and materials provided on the Page at any time and without notice. The developer(s) also reserve the right to terminate access to the Page or to any code or materials provided on the Page at any time and without notice.

### 6.2 Copyright Notice: ###
Copyright (c) 2023 Idris Awad. All rights reserved. Any code or other materials provided on this Page are the property of the developer(s) and are protected by copyright and other intellectual property laws. Unauthorized use, reproduction, distribution, or creation of derivative works is prohibited.

Please note that using, reproducing or distributing the code or materials provided on this Page without proper attribution and without obtaining express permission from the developer(s) may result in copyright infringement and legal action being taken against you.

By accessing and using this Page, you acknowledge and agree to the terms of this legal disclaimer, policy and copyright notice.


