# Results Management AI

Hello! Welcome to our Results Management container! This is specifically designed to be a standardized environment you can use to get our Results Management tool up and running quickly. Using this tool negates the need to have the fun experience of trying to recreate our exact environment - we've built and packaged it up for you! Feel free to use as is, or fiddle with it to suit your own needs. However, be warned, there is some quirkiness with `code server` and stable configurations, so know that some altering the extensions may break things. Good news though, if you ran this with the `--rm` flag, you'll start from a clean slate every time.

## Get Started!
There's a couple ways you can tackle this, it's a bit of a choose your own adventure:
- Look at our online [documentation](https://mozzilab.github.io/NM_Radiology_AI), or online demo walkthroughs, for [phase 01](https://mozzilab.github.io/NM_Radiology_AI/phase01/phase01.html) and [phase 02](https://mozzilab.github.io/NM_Radiology_AI/phase02/phase02.html)
- Explore the demo jupyter notebooks for either the `phase 01` [demo](didact://?commandId=vscode.open&projectFilePath=code/examples/phase01/demo.ipynb) or, for the latest work, the `phase 02` [pretraining](didact://?commandId=vscode.open&projectFilePath=code/examples/phase02/demo_pretrain.ipynb) and [fine-tuning](didact://?commandId=vscode.open&projectFilePath=code/examples/phase02/demo_finetune.ipynb) demos. These notebooks provide more insight and sample code outputs for a small, sample dataset. These demos are based on the underyling source code.

- Look at the raw python source code (`/workspace/code/src/nmrezman`)
- Try training on your own data with the code as-is (try running it as a module). To run as a module, change your directory to `/workspace/code/src` and then run the code as `python -m nmrezman.phase01.train.train_findings <args>`
- If the code needs to be modified to suit your needs, you can change the source code

>  ℹ️ &ensp; | &ensp; You must have the following prequsites downloaded and setup before running the Phase 01 models. Do this by either manual downloading and extracting in the workspace or using `wget` in the cli to accomplish the same. The demo notebooks have sections detailing how to do this as well. We suggest you store these in the `/workspace/data/` (e.g., this would then serve as your `/path/to/data`).
> 
> | Model (Phase 01)                   | Downloads                  |                                                                                                                                 
> | ---------------------------------- | -------------------------- |
> | Findings vs No Finding Model       | [GloVe pretrained word vectors: glove.6B.300d.txt](https://nlp.stanford.edu/data/glove.6B.zip) |
> | Lung vs Adrenal Findings Model     | [BioWordVec word vectors: BioWordVec_PubMed_MIMICIII_d200.bin](https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioWordVec_PubMed_MIMICIII_d200.bin) |

## Some quick VS code tips
### Clone an existing repo
1. [Start a new Terminal](didact://?commandId=workbench.action.terminal.new "Start a new Terminal"), which is also achieved via <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>`</kbd>
1. Enter your git credentials in terminal:
    ```
    git config --global user.name <user_name>
    git config --global user.email <email>
    ```
1. [Git Clone your Repo](didact://?commandId=git.clone "Git Clone your Repo"), which is also achieved via opening the Command Palette (see [below](#useful-tips)) and searching for "Clone"
1. [Open Git Graph to view your Git History](didact://?commandId=git-graph.view)

## Useful Tips
- [Open the Command Palette](didact://?commandId=workbench.action.showCommands "Open the Command Palette"), which is also achieved via <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>P</kbd>
- [Change Settings via UI](didact://?commandId=workbench.action.openSettings "Change Settings via UI")
- [Change the Remote's Settings (.json)](didact://?commandId=workbench.action.openRemoteSettings)

### Info
- Author: HIT Lab