

# Welcome to pyrfu

PYRFU is a software based on the IRFU-MATLAB library to work with space data, particularly the Magnetospheric MultiScale (MMS) mission. 


# Instalation
pip install --index-url https://test.pypi.org/project/ --no-deps pyrf 

# Usage
To import generic space plasma physics functions
```python
from pyrfu import pyrf
```

To import functions specific to MMS mission
```python
from pyrfu import mms
```

To import plotting functions
```python
from pyrfu import plot as pltrf
```

# Credits 
This software was developped by Louis RICHARD (louisr@irfu.se) based on the IRFU-MATLAB library.


# Acknowloedgement
Please use the following to acknowledge use of pyrfu in your publications:
Data analysis was performed using the pyrfu analysis package available at https://github.com/louis-richard/irfu-python
