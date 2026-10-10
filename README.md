#  🤖 AI-Text-Generator

🤖 AI Text Generator is a Streamlit-based application powered by a fine-tuned DistilGPT-2 language model. It generates meaningful text continuations from user prompts using NLP and transformer technology. The app offers customizable generation settings, live prompt preview, generated text statistics, and text download functionality.

# 🤖 AI Text Generator — TextNova AI

## 📌 Project Overview

AI Text Generator is an NLP-based application that generates meaningful text continuations from user-provided prompts. It uses a fine-tuned DistilGPT-2 causal language model to predict and generate the next sequence of words based on the input text. The application is built with Streamlit and provides an interactive dashboard with customizable generation settings, live prompt preview, text generation statistics, and downloadable output.

## 🎯 Objectives

- Generate meaningful text from user-provided prompts.
- Apply Natural Language Processing (NLP) and transformer-based language modeling.
- Understand causal language modeling and next-token prediction.
- Provide an interactive and user-friendly text generation interface.
- Allow users to customize text generation parameters.

## ✨ Key Features

- 🤖 AI-powered text generation
- 🧠 Fine-tuned DistilGPT-2 language model
- ✍️ Custom prompt input
- 👀 Live prompt preview
- ⚙️ Adjustable text generation settings
- 🌡️ Temperature control
- 🔝 Top-K and Top-P sampling
- 📝 Customizable maximum new tokens
- 📊 Input and output token statistics
- ⏱️ Text generation time tracking
- ⬇️ Download generated text
- 🗑️ Clear and reset functionality
- 💻 GPU and CPU device support
- 🎨 Interactive Streamlit dashboard

## 🏗️ Model Architecture

- 🧠 **Model Type:** Causal Language Model
- 🤖 **Base Model:** DistilGPT-2
- 🔧 **Model Framework:** Hugging Face Transformers
- ✍️ **Task:** Text Generation
- 📚 **Training Dataset:** ELI5 Category Dataset
- 📊 **Dataset Samples:** First 5,000 dataset examples selected
- 🎯 **Training Approach:** Fine-tuning
- 🔤 **Tokenization:** GPT-2 Tokenizer
- 🧩 **Training Sequence Length:** 128 tokens
- 📝 **Output:** Generated text continuation

## 🔄 Project Workflow

1. 📥 Load the ELI5 Category dataset.
2. ✂️ Split the dataset into training and testing sets.
3. 🔤 Load the pretrained DistilGPT-2 tokenizer.
4. 🧹 Tokenize and preprocess the text data.
5. 🧩 Group token sequences into blocks of 128 tokens.
6. 🧠 Load the pretrained DistilGPT-2 causal language model.
7. 🚀 Fine-tune the model using the Hugging Face Trainer.
8. 📊 Evaluate the model using evaluation loss and perplexity.
9. 🤗 Save and upload the trained model to Hugging Face.
10. 🌐 Load the fine-tuned model into the Streamlit application.
11. ✍️ Accept a text prompt from the user.
12. ⚙️ Generate a text continuation using configurable sampling parameters.
13. 📈 Display, analyze, and download the generated text.

## ⚙️ Text Generation Parameters

| Parameter | Description |
|---|---|
| Max New Tokens | Controls the maximum number of new tokens to generate. |
| Temperature | Controls randomness in text generation. |
| Top-K | Limits token selection to the top K candidates. |
| Top-P | Selects candidate tokens using cumulative probability. |
| Repetition Penalty | Helps reduce repeated phrases. |
| No Repeat N-Gram | Helps prevent repeated n-gram sequences. |

## 🛠️ Technologies Used

- 🐍 Python
- 🤖 Artificial Intelligence
- 📈 Machine Learning
- 💬 Natural Language Processing (NLP)
- 🧠 Deep Learning
- 🔄 Transformer Models
- ✨ DistilGPT-2
- 🤗 Hugging Face Transformers
- 📚 Hugging Face Datasets
- 🔥 PyTorch
- 🔤 Tokenization
- 📝 Causal Language Modeling
- ☁️ Google Colab
- 🌐 Streamlit
## 💡 Example

**Input Prompt:**

The human immune system protects the body by

**Generated Output:**

The application generates a continuation based on the given prompt and the patterns learned during model training.

*Note: The actual output varies depending on the model, prompt, and generation settings.*

## 📊 Evaluation

The model training workflow uses evaluation loss and perplexity to assess language modeling performance.

- **Evaluation Loss:** Measures the model's prediction error on the evaluation dataset.
- **Perplexity:** Estimates how well the model predicts the next tokens. Lower perplexity generally indicates better predictive performance on the evaluated data.

The Streamlit application also displays prompt token count, output token count, and generation time.

## 💡 Applications

- ✍️ AI-assisted writing
- 📝 Text completion
- 🎓 Educational content generation
- 🎨 Creative writing assistance
- 📄 Content drafting
- 🧪 NLP experimentation
- 📚 Language model research and learning

## 🔮 Future Scope

- 🚀 Integrate larger and more capable language models.
- 🧠 Improve text coherence and contextual relevance.
- ✍️ Add multiple text generation styles.
- 🌐 Support multilingual text generation.
- 💾 Introduce prompt history and saved generations.
- 🎛️ Add controls for output length and writing style.
- 📊 Improve model evaluation with additional metrics.
- ⚡ Optimize inference speed and memory consumption.
- 🔗 Add an API for integration with other applications.

## ⚠️ Limitations

- ⚠️ Generated text may contain inaccurate or repetitive information.
- 📚 Output quality depends on the training data and input prompt.
- 🧠 The model may not consistently maintain long-range context.
- ✅ Generated text should be reviewed before use in important applications.
- ⚡ Generation speed depends on available hardware and computational resources.

## 🏁 Conclusion

AI Text Generator demonstrates how a pretrained transformer language model can be fine-tuned for causal language modeling and deployed through an interactive Streamlit interface. The project combines NLP, tokenization, deep learning, and text generation to create a practical application for experimenting with AI-generated text.
