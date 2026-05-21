# My-project
#Diagnosing Retrieval-Augmented Generation: A Joint Evaluation of Faithfulness and Retrieval Utility

This project has implemented a dual evaluation framework for the Retrieval Augmented Generation (RAG) system:
-SEPER: Measuring the utility changes brought by retrieval based on semantic perplexity, fully aligned with paper metrics
-RAGAS: Framework based standard indicator evaluation (loyalty, answer relevance, contextual accuracy, etc.)

##Environment configuration

```bash
#Core Dependency
pip install transformers==4.43.1
pip install sentencepiece
pip install ragas==0.4.3
pip install langchain==1.2.13
pip install sentence_transformers
pip install seaborn
pip install faiss-gpu
```

##Project Structure

```
##Project Structure

```
∝ - download.py # Deploy the required models
∝ - ragas_evaluation_main.py # RAGAS evaluation main script
∝ - seper_evaluation_main.py # SEPER evaluation main script
∝ - comparisonAnalyzer_visualization_main.py # SEPER and RAGAS Comparison Visualization

##Usage method

### 1. Run SEPER evaluation
```bash
python seper_evaluation_main.py
```
### 2. Run RAGAS evaluation
```bash
python ragas_evaluation_main.py
```
### 3. Generate comparative analysis and visualization
```bash
python comparisonAnalyzer_visualization_main.py
```
