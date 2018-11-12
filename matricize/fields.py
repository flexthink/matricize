from typing import Sequence, Dict
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

_field_map = defaultdict(lambda: {})

@dataclass
class Field:
    """
    A base class for all field definitions. Each field definition is a descriptor that defines how to
    represent a given piece of data in natural Python and how to convert it to a dense matrix
    representation
    """
    output_name: str = 'default'

    def __set_name__(self, owner, name):
        _field_map[owner][name] = self
        self.name = name

    def __get__(self, obj, type=None):
        return obj.__dict__[self.name]

    def __set__(self, obj, value):
        obj.__dict__[self.name] = value

    def matricize(self, items: Sequence, target: Dict):
        """
        Converts this field to its matrix representation in the target
        dictionary

        :param items: a collection of raw data objects
        :param target: a dictionary of {'name': <numpy array>} to which the field's matrix
                       or tensor representation will be output
        """
        raise NotImplementedError()


@dataclass(init=False)
class Categorical(Field):
    """
    Represents a categorical (enumerated) field where
    values are chosen from among a finite set of options. Categorical fields will be one-hot
    encoded
    """
    categories: Sequence

    def __init__(self, categories: Sequence, output_name='default'):
        super().__init__(output_name=output_name)
        if not categories:
            raise ValueError("Categories are required")
        self.categories = categories
        self._category_map = {category: idx for idx, category in enumerate(categories)}
        self._reference_matrix = np.eye(len(categories))

    def matricize(self, items, target):
        indices = np.array([self._category_map[item.__dict__[self.name]] for item in items], dtype=np.int)
        target[self.output_name] = self._reference_matrix[indices]


@dataclass
class Boolean(Field):
    """
    Represents a Boolean (True/False) field, encoded as 1=True, 0=False. Multiple Boolean fields
    can be concatenated together into a single matrix
    """
    index: int = 0

    def matricize(self, items, target):
        target[self.output_name][:, self.index] = [
            1 if obj.__dict__[self.name] else 0 for obj in items]


@dataclass
class Number(Field):
    """
    Represents a regular numeric field. A single data object may contain multiple numeric fields,
    which can be gathered into a more compact presentation where each numeric field corresponds to 
    an element in a vector
    """
    index: int = 0

    def matricize(self, items, target):
        target[self.output_name][:, self.index] = [
            obj.__dict__[self.name] for obj in items]
    

def get_fields(source):
    """
    Returns all fields registered for the specified object or class

    :param source: An object or a class
    :rtype: dict
    """
    if not isinstance(source, type):
        source = source.__class__
    
    fields = _field_map.get(source)
    if not fields:
        raise ValueError(f"Source '{source}' is not registered'")
    return fields


def get_field(source, name):
    """
    Returns the field instance registered for the specified field name 
    for a given class

    :param source: An object or a class
    :rtype: Field
    """
    fields = get_fields(source)
    field = fields.get(name)
    if not field:
        raise ValueError(f"Field '{name} is not registered for the source")
    return field