# Run the VisRAG pipeline

This directory contains code for building an index from custom PDF files and running the VisRAG retrieval and generation pipeline on it.

First run `build_index.py` which will prompt you to specify a path to your PDF files and a path for storing the index. Then run `answer.py` to query your knowledge base.

Note that to successfully run the pipeline you may need a GPU with ~40GB memory.
