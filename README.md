# MOT ([Mouth of Truth](https://en.wikipedia.org/wiki/Bocca_della_Verit%C3%A0))
To open the repo of MOT.

## 🛠️ Prerequisites

Before running the project, ensure you have the following installed:

* **Python** >= 3.8
* **Soufflé** (For Datalog logic analysis)
* **Java / Go** (For specific chain clients, if applicable)

---

## 📂 Project Structure

```text
MOT/
├── code/                  # Main source code
│   ├── py/                # python code  
│   ├── dl/                # Datalog code
├── data/                  # Dataset of MOT
└── README.md              # Project documentation

```
---

🚀 Quick Start

### Run Code

```bibtex
cd ./code/py/xxx.py
python xxx.py

cd ./code/dl/yyy.dl
souffle -F TxSM_facts -D yyy.dl

```


![MOT](MOT.png)
