import os
from unittest import TestCase

import tensorflow.keras as keras
from pydantic import BaseModel
from testfixtures import TempDirectory

from mlnext import io


class Config(BaseModel):
    name: str


class TestConfig(TestCase):

    def setUp(self):
        self.d = TempDirectory()

    def tearDown(self):
        self.d.cleanup()

    def test_save_config(self):
        data = Config(name='test')
        name = 'test.yaml'

        io.save_config(config=data, folder=self.d.path, name=name)

        self.assertTrue(os.path.isfile(os.path.join(self.d.path, name)))

    def test_save_config_no_ext(self):

        data = Config(name='test')
        name = 'test1'

        io.save_config(config=data, folder=self.d.path, name=name)

        self.assertTrue(os.path.isfile(
            os.path.join(self.d.path, f'{name}.yaml')))

    def test_save_config_invalid_dir(self):

        with self.assertRaises(ValueError):
            io.save_config(config={}, folder='', name='')


class TestJson(TestCase):
    def setUp(self):
        self.d = TempDirectory()

    def tearDown(self):
        self.d.cleanup()

    def test_save_json(self):

        data = {'test': [0, 1, 2]}
        name = 'test.json'

        io.save_json(data=data, folder=self.d.path, name=name)

        self.assertTrue(os.path.isfile(os.path.join(self.d.path, name)))

    def test_save_json_no_ext(self):

        data = {'test': [0, 1, 2]}
        name = 'test1'

        io.save_json(data=data, folder=self.d.path, name=name)

        self.assertTrue(os.path.isfile(
            os.path.join(self.d.path, f'{name}.json')))

    def test_save_json_invalid_dir(self):

        with self.assertRaises(ValueError):
            io.save_json(data={}, folder='', name='')

    def test_load_json(self):

        data = {'test': 'test'}
        name = 'test.json'

        io.save_json(data=data, folder=self.d.path, name=name)
        result = io.load_json(os.path.join(self.d.path, name))

        self.assertDictEqual(result, data)

    def test_load_json_invalid_path(self):

        with self.assertRaises(FileNotFoundError):
            io.load_json(os.path.join(self.d.path))


class TestYaml(TestCase):
    def setUp(self):
        self.d = TempDirectory()

    def tearDown(self):
        self.d.cleanup()

    def test_save_yaml(self):
        data = {'test': [0, 1, 2]}
        name = 'test.yaml'

        io.save_yaml(data=data, folder=self.d.path, name=name)

        self.assertTrue(os.path.isfile(os.path.join(self.d.path, name)))

    def test_save_yaml_no_ext(self):

        data = {'test': [0, 1, 2]}
        name = 'test1'

        io.save_yaml(data=data, folder=self.d.path, name=name)

        self.assertTrue(os.path.isfile(
            os.path.join(self.d.path, f'{name}.yaml')))

    def test_save_yaml_invalid_dir(self):

        with self.assertRaises(ValueError):
            io.save_yaml(data={}, folder='', name='')

    def test_load_yaml(self):

        data = {'test': 'test'}
        name = 'test.yaml'

        io.save_yaml(data=data, folder=self.d.path, name=name)
        result = io.load_yaml(os.path.join(self.d.path, name))

        self.assertDictEqual(result, data)

    def test_load_yaml_invalid_path(self):

        with self.assertRaises(FileNotFoundError):
            io.load_yaml(os.path.join(self.d.path))


class TestLoadModel(TestCase):
    def setUp(self):
        self.d = TempDirectory()

    def tearDown(self):
        self.d.cleanup()

    def test_load_model_invalid_dir(self):
        with self.assertRaises(ValueError):
            io.load_model('./abc')

    def test_load_model(self):
        m = keras.Sequential([keras.layers.Input(10), keras.layers.Dense(1)])
        m.save(path := os.path.join(self.d.path, 'model'))

        result = io.load_model(path)

        self.assertIsInstance(result, keras.Model)
