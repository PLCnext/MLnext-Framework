import os
from unittest import TestCase

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

    def test_save_invalid_ext(self):
        data = {'test': [0, 1, 2]}
        name = 'test.txt'

        with self.assertRaises(ValueError):
            io.save_json(data=data, folder=self.d.path, name=name)

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

    def test_save_yml(self):
        data = {'test': [0, 1, 2]}
        name = 'test.yml'

        io.save_yaml(data=data, folder=self.d.path, name=name)

        self.assertTrue(os.path.isfile(os.path.join(self.d.path, name)))

    def test_save_invalid_ext(self):
        data = {'test': [0, 1, 2]}
        name = 'test.txt'

        with self.assertRaises(ValueError):
            io.save_yaml(data=data, folder=self.d.path, name=name)

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


class TestLoad(TestCase):

    def setUp(self):
        self.d = TempDirectory()
        self.data = {'test': [0, 1, 2]}

    def tearDown(self):
        self.d.cleanup()

    def test_yaml(self):
        name = 'test.yaml'
        io.save_yaml(data=self.data, name=name, folder=self.d.path)

        result = io.load(path=os.path.join(self.d.path, name))

        self.assertDictEqual(result, self.data)

    def test_yml(self):
        name = 'test.yml'
        io.save_yaml(data=self.data, name=name, folder=self.d.path)

        result = io.load(path=os.path.join(self.d.path, name))

        self.assertDictEqual(result, self.data)

    def test_json(self):
        name = 'test.json'
        io.save_json(data=self.data, name=name, folder=self.d.path)

        result = io.load(path=os.path.join(self.d.path, name))

        self.assertDictEqual(result, self.data)

    def test_invalid_ext(self):
        name = 'test.abc'

        with self.assertRaises(ValueError):
            io.load(name)


class TestList(TestCase):

    def setUp(self):
        self.d = TempDirectory()
        self.d.makedir('tasks')
        self.d.makedir('models')

        self.data = {'test': [1, 2, 3]}
        self.tasks = os.path.join(self.d.path, 'tasks')
        io.save_json(data=self.data, name='test.json', folder=self.tasks)
        io.save_json(data=self.data, name='foo.json', folder=self.tasks)

    def tearDown(self):
        self.d.cleanup()

    def test_get_folders(self):
        expected = ['models', 'tasks']

        result = io.get_folders(self.d.path)

        self.assertCountEqual(result, expected)

    def test_get_folders_full_path(self):
        expected = [os.path.join(self.d.path, p) for p in ['models', 'tasks']]

        result = io.get_folders(self.d.path, absolute=True)

        self.assertCountEqual(result, expected)

    def test_get_folders_filter(self):
        expected = ['models']

        result = io.get_folders(self.d.path, filter='m')

        self.assertCountEqual(result, expected)

    def test_get_folders_invalid_path(self):
        with self.assertRaises(ValueError):
            io.get_folders(os.path.join(self.tasks, 'test.json'))

    def test_get_files(self):
        expected = ['test.json', 'foo.json']

        result = io.get_files(path=self.tasks, ext='json')

        self.assertCountEqual(result, expected)

    def test_get_files_filter(self):
        expected = ['foo.json']

        result = io.get_files(path=self.tasks, ext='json', name='foo')

        self.assertCountEqual(result, expected)

    def test_get_files_full_path(self):
        expected = [os.path.join(self.tasks, p)
                    for p in ['test.json', 'foo.json']]

        result = io.get_files(path=self.tasks, ext='json',
                              absolute=True)

        self.assertCountEqual(result, expected)

    def test_get_files_invalid_path(self):
        with self.assertRaises(ValueError):
            io.get_files(path=os.path.join(self.tasks, 'test.json'),
                         ext='json')
