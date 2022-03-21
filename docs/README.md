# How to Build the Docs

1. Install sphinx
    ```
    pip install sphinx sphinx-copybutton sphinx_rtd_theme sphinx-panels nbsphinx
    ```
1. Pip install the `nmrezman` package (command below is if the repo is local)
    ```
    pip install /path/to/repo//NM_Radiology_AI
    ```
1. In docs dir, create html files
    ```
    make html
    ```
1. To start from scratch, clean the build files
    ```
    make clean
    ```
1. The docs can be previewed locally at `localhost:<port_number>` via
    ```
    cd /path/to/repo/docs/build/html
    python -m http.server <port_number>
    ```
