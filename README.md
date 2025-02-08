# Dream Interpretation using GPT-2 and T5

## Introduction
This project explores the use of NLP models for interpreting dreams based on 2 datasets of dreams descriptions. The goal is to analyze and compare the effectiveness of 4 models in generating dream interpretations.
This project aims to build an AI dream interpretor by leveraging LLMs trained on dream datasets.

**Key Features:**
- **GPT-2 and T5 Models**: Two transformer models trained to generate dream interpretations.
- **Add other models**
- **Automated Evaluation**: Performance comparison using **BLEU, ROUGE, BERTScore, and Perplexity**.
- **Test and Validation Analysis**: Insights about how well the models generalize to unseen dream data.

---

## Dependencies
Ensure the following dependencies are installed before running the code:

- Python > 3.6
- Pandas 
- NumPy 
- Regular Expressions (re) 
- Datetime 
- Scikit-learn 
- Hugging Face Transformers (pip install transformers)
- Datasets (pip install datasets)
- PyTorch 
- NLTK 
- OpenCV 
- pytorch-gan-metrics 
- CUDA (Optional) for faster training on NVIDIA GPU
---

## Implementation

### **1. Download the Dataset**
The dream dataset used in this project is available on **Dryad**:  
🔗 [Dataset Link](https://datadryad.org/stash/dataset/doi:10.5061/dryad.qbzkh18fr)

After downloading, preprocess the dataset to prepare it for training.

---

### **2. Train the Models**
Train GPT-2 and T5 models using the preprocessed dataset:

```bash
python train.py --model gpt2 --epochs 10
python train.py --model t5 --epochs 10
```

This will fine-tune the models on dream interpretation data.

---

### **3. Evaluate the Models**
Run the evaluation script to compare model performance:

```bash
python evaluate.py --test_dataset test_data.json
```

Metrics used:
- **BLEU Score** (Measures similarity to reference interpretations)
- **ROUGE Scores** (Measures n-gram overlap)
- **BERTScore** (Semantic similarity)
- **Perplexity** (Measures model confidence)

---

## **Results**

### **Test Dataset Performance**
| Metric               | GPT-2  | T5    |
|----------------------|--------|-------|
| **BLEU Score**      | 0.0449 | 0.9148 |
| **ROUGE-1**        | 0.1678 | 0.9229 |
| **ROUGE-2**        | 0.0545 | 0.8778 |
| **ROUGE-L**        | 0.1304 | 0.9201 |
| **Perplexity**      | 1.334e11 | 1.0665 |
| **BERTScore (F1)** | 0.7913 | 0.9537 |

📌 **Conclusion:**  
- **T5 outperforms GPT-2 significantly** in accuracy and fluency.
- **Lower Perplexity in T5** means it generates more confident interpretations.

---

### **Validation Dataset Performance**
| Metric               | GPT-2  | T5    |
|----------------------|--------|-------|
| **BLEU Score**      | 0.0000 | 0.0000 |
| **ROUGE-1**        | 0.0333 | 0.0626 |
| **ROUGE-2**        | 0.0000 | 0.0036 |
| **ROUGE-L**        | 0.0287 | 0.0518 |
| **Perplexity**      | 1.0974 | 1.0639e6 |
| **BERTScore (F1)** | 0.7897 | 0.7984 |

📌 **Key Takeaways:**  
- **Performance drops significantly on validation data**, suggesting **overfitting** or challenges in dream interpretation generalization.
- **T5 still performs better than GPT-2**, but the gap is smaller.

---

## **Visualizations**
The following plots illustrate the comparative performance of the two models:

![Performance Comparison](results/performance_comparison.png)

---

## **Next Steps & Future Improvements**
- **Data Augmentation**: Improve generalization with **more diverse dream datasets**.
- **Fine-tuning with Larger Models**: Explore **T5-large** or **GPT-3** for better results.
- **Hybrid Approach**: Combine GPT-2’s fluency with T5’s accuracy for enhanced performance.

---

## **Citation**
If you use this project, please cite the original dataset:
```
@dataset{dream_dataset_2025,
  title={Dream Interpretations Dataset},
  author={Author Name},
  year={2025},
  publisher={Dryad},
  doi={10.5061/dryad.qbzkh18fr}
}
```

---

## **Contact**
For any issues or inquiries, feel free to open an issue or reach out via **GitHub Discussions**.

---

## **Acknowledgments**
Special thanks to the authors of the original dataset and the open-source contributors of **Hugging Face Transformers**.

🚀 **Happy Dream Interpretation!**
