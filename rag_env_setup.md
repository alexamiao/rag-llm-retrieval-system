# 🔧 Create your RAG Python Environment

---

## ✏️ Steps

1. 
    ```bash
    conda create -n rag_313 python=3.13
    ```

2. 
    ```bash
    conda activate rag_313
    ```

3. 
    ```bash
    conda install ipykernel sentence-transformers chromadb requests numpy pandas
    ```

## 📜 Note

Use rag_313 as your kernel for you project moving forward. 

- If you need to add more libraries. (Ensure you activate environment before installing)

    - `conda install <library name>`