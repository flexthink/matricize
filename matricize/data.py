from .fields import get_fields
from itertools import groupby
import numpy as np

def _get_output_name(field):
    return field.output_name

def _build_matrix(items, target, output_name, fields, count):
    # TODO: Add support for higher dimensionality
    fields = list(fields)
    target[output_name] = np.zeros((count, len(fields)))
    for field in fields:
        field.matricize(items, target)

#TODO: Add support for batches
def matricize(items):
    """
    Converts a collection of complex data elements to a dictionary of
    matrices or tensors

    :param items: A collection of data objects of the same type, for which
                  one or more fields have been defined using Matricize
                  descriptors
    """
    if not items:
        raise ValueError("No items found")
    count = len(items)
    item = items[0]
    fields = get_fields(item)
    if not fields:
        raise ValueError("Unable to matricize the collection because no "
        "Matricize fields have been defined for the objects in it")
    fields = sorted(fields.values(), key=_get_output_name)
    grouped_fields = groupby(fields, _get_output_name)
    result = {}
    for key, fields in grouped_fields:
        _build_matrix(items, result, key, fields, count)
    return result
    
    

