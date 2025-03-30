# Faster way of doing Tf-IDF

**CPU Libraries (scikit-learn)**: Used for **TF-IDF** on CPU.

**GPU Libraries (cuDF, cuML, CuPy)**: Used for **accelerated TF-IDF and similarity search**

## **Text Vectorization (TF-IDF)**

There are **two methods**:

1. **Using `HashingVectorizer` + `TfidfTransformer`**
2. **Using `TfidfVectorizer`**
- **`HashingVectorizer`**: Converts text into high-dimensional vectors **(bag-of-words)**
- **`TfidfTransformer`**: Converts these counts into **TF-IDF scores**.

**CPU Search**

- Converts **query into a TF-IDF vector**.
- Computes **cosine similarity** with all documents.
- **Returns top `N` most similar results**.

### **Optimized GPU Search using `NearestNeighbors`**

Instead of **manual matrix multiplication**, use **cuML’s NearestNeighbors (faster)**

- **Normalizes TF-IDF vectors** for accuracy.
- Uses **GPU-optimized Nearest Neighbors** for **fast similarity search**.

- Compares **CPU vs GPU performance** for:
    - **Data loading**
    - **TF-IDF vectorization**
    - **Similarity search**
- Computes **speedup factor**

**Format to save the tf-idf vectors**
