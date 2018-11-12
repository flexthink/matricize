import unittest
import numpy as np
from dataclasses import dataclass

from matricize.fields import Number, Categorical, Boolean, get_field, get_fields

@dataclass
class SampleObject:
    kind: str = Categorical(output_name='kind', categories=['vehicle', 'building', 'person'])
    x: float = Number(output_name='position', index=0)
    y: float = Number(output_name='position', index=1)


@dataclass
class Person:
    name: str
    bachelors: bool = Boolean(output_name='state', index=0)
    masters: bool = Boolean(output_name='state', index=1)
    phd: bool = Boolean(output_name='state', index=2)
    employed: bool = Boolean(output_name='state', index=3)
    married: bool = Boolean(output_name='state', index=4)


class FieldsTest(unittest.TestCase):
    def test_get_fields_by_class(self): 
        fields = get_fields(SampleObject)
        self.assertIn('x', fields)
        self.assertIn('y', fields)
        self.assertIsInstance(fields['x'], Number)
        self.assertIsInstance(fields['y'], Number)
        
    def test_get_fields_by_object(self): 
        obj = SampleObject(kind='vehicle', x=1.0, y=2.0)
        fields = get_fields(obj)
        self.assertIn('x', fields)
        self.assertIn('y', fields)
        self.assertIsInstance(fields['x'], Number)
        self.assertIsInstance(fields['y'], Number)

    def test_get_field_by_class(self):
        field = get_field(SampleObject, 'x')
        self.assertIsNotNone(field)
        self.assertIsInstance(field, Number)

    def test_get_field_by_object(self):
        obj = SampleObject(kind='vehicle', x=1.0, y=2.0)
        field = get_field(obj, 'x')
        self.assertIsNotNone(field)
        self.assertIsInstance(field, Number)


class NumberTest(unittest.TestCase):
    def test_matricize(self):
        items = [
            SampleObject(kind='vehicle', x=1.0, y=5.0),
            SampleObject(kind='vehicle', x=3.0, y=4.0),
            SampleObject(kind='vehicle', x=4.0, y=7.0)
        ]
        field = get_field(SampleObject, 'x')
        target = {'position': np.zeros((3, 2))}
        field.matricize(items, target)
        reference_positions = np.array([
            [1., 0.],
            [3., 0.], 
            [4., 0.]])
        np.testing.assert_almost_equal(reference_positions, target['position'])
        field = get_field(SampleObject, 'y')
        field.matricize(items, target)
        reference_positions = np.array([
            [1., 5.],
            [3., 4.], 
            [4., 7.]])
        np.testing.assert_almost_equal(reference_positions, target['position'])


class CategoricalTest(unittest.TestCase):
    def test_matricize(self):
        items = [
            SampleObject(kind='person', x=1.0, y=5.0),
            SampleObject(kind='vehicle', x=3.0, y=4.0),
            SampleObject(kind='building', x=4.0, y=7.0)
        ]
        reference = np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ])
        target = {'kind': np.zeros((3, 3), dtype=np.int)}
        field = get_field(SampleObject, 'kind')
        field.matricize(items, target)
        np.testing.assert_equal(reference, target['kind'])


class BooleanTest(unittest.TestCase):
    def test_matricize(self):
        items = [
            Person(name='John Doe', bachelors=True, masters=False, phd=False,
                employed=True, married=True),
            Person(name='Jane McDonald', bachelors=True, masters=True,
                phd=True, employed=True, married=False),
            Person(name='Michael Smith', bachelors=True, masters=False,
                phd=False, employed=False, married=True),
        ]
        reference = np.array([
            [1, 0, 0, 1, 1],
            [1, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
        ], dtype=np.int)
        target = {'state': np.zeros((3, 5), dtype=np.int)}
        fields = get_fields(Person)
        for field in fields.values():
            field.matricize(items, target)
        np.testing.assert_equal(reference, target['state'])
