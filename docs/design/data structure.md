# Data structure design
To make the python API more native, there are several builtin data structures:

- `List`, similar to `list` in python
- `Dict`, similar to `dict` in python
- `Set`, similar to `set` in python
- `String`, similar to `str` in python

User can only use the data structure above in the Python kernel.
But use the python native data structures and replace during AOT should be a better idea, which might be supported in the future.
