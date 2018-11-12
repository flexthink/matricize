import unittest
import numpy as np
from dataclasses import dataclass

from matricize.fields import Number, Categorical, Boolean
from matricize.data import matricize

@dataclass
class SampleObject:
    kind: str = Categorical(output_name='kind',
                       categories=['vehicle', 'building', 'person', 'water', 'object'])

    x: float = Number(output_name='position', index=0)
    y: float = Number(output_name='position', index=1)

    height: float = Number(output_name='dimensions', index=0)
    width: float = Number(output_name='dimensions', index=1)
    weight: float = Number(output_name='dimensions', index=2)

    verified: bool = Boolean(output_name='parameters', index=0)
    local: bool = Boolean(output_name='parameters', index=1)


class DataTest(unittest.TestCase):
    def test_matricize(self):
        items = [
            SampleObject(kind='person', x=1.0, y=2.0, height=6., width=12., weight=160.,
                         verified=False, local=False),
            SampleObject(kind='building', x=12.0, y=20.0, height=100., width=12.,
                         weight=10000., verified=False, local=True),
            SampleObject(kind='water', x=50.0, y=50.0, height=100., width=200., weight=20000.,
                         verified=True, local=True),
            SampleObject(kind='object', x=25.0, y=30.0, height=100., width=200., weight=20000.,
                         verified=True, local=False),
        ]
        representation = matricize(items)
        self.assertIsNotNone(representation)
        reference = {
            'kind': np.array([
                [0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
            ], dtype=np.int),
            'position': np.array([
                [1., 2.],
                [12., 20.],
                [50., 50.],
                [25., 30.],
            ]),
            'parameters': np.array([
                [0, 0],
                [0, 1],
                [1, 1],
                [1, 0],
            ], dtype=np.int)
        }
        for key, reference_value in reference.items():
            self.assertIn(key, representation)
            np.testing.assert_almost_equal(reference_value, representation[key])
