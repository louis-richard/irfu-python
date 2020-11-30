##############
Quick overview
##############

Here are some quick examples of what you can do with :py:`pyrfu`. Everything is explained in much more detail in the rest of the
documentation.


Define a time interval
----------------------

Time intervals are defined as list of two string where the strings are the begining and the end
time of the interval in isot format.
You can make a DataArray from scratch by supplying data in the form of a numpy
array or list, with optional *dimensions* and *coordinates*:

.. ipython:: python

    tint = ["2015-10-30T05:15:40.000", "2015-10-30T05:15:55.000"]

In this case, we have generated a 2D array, assigned the names *x* and *y* to the two dimensions respectively and associated two *coordinate labels* '10' and '20' with the two locations along the x dimension. If you supply a pandas :py:class:`~pandas.Series` or :py:class:`~pandas.DataFrame`, metadata is copied directly:

.. ipython:: python

    xr.DataArray(pd.Series(range(3), index=list("abc"), name="foo"))

Here are the key properties for a ``DataArray``:

.. ipython:: python

    # like in pandas, values is a numpy array that you can modify in-place
    data.values
    data.dims
    data.coords
    # you can use this dictionary to store arbitrary metadata
    data.attrs