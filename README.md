

# pyRFU

pyRFU is a software based on the IRFU-MATLAB library to work with space data, particularly the Magnetospheric MultiScale (MMS) mission. 


# Instalation
pyRFU supports Windows, macOS and Linux. pyRFU uses TestPyPI a separate instance of the Python Package index to not affect the real index. To get started, install the pyrfu package using TestPyPI:

```python
pip install --index-url https://test.pypi.org/project/ --no-deps pyrfu 
```


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
