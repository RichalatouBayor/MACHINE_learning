#!/usr/bin/env python3

# Libs unittest
import unittest
from unittest.mock import patch
from unittest.mock import Mock

# Utils libs
import os
import shutil
import pandas as pd
import numpy as np
from ynov import utils

# Disable logging
import logging
logging.disable(logging.CRITICAL)


def remove_dir(path):
    if os.path.isdir(path): shutil.rmtree(path)


class UtilsTests(unittest.TestCase):
    '''Main class to test all functions in utils.py'''
    # On evite les prints de tqdm
    pd.Series.progress_apply = pd.Series.apply


    def setUp(self):
        '''SetUp fonction'''
        # On se place dans le bon répertoire
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)


    @patch('logging.Logger._log')
    def test01_read_csv(self, PrintMockLog):
        '''Test de la fonction utils.read_csv'''

        # Arguments df1
        path1 = './test_dataset.csv'
        sep1 = ','
        df1_shape_expected = (100, 12)
        first_line1_expected = None

        # Arguments df2
        path2 = './test_dataset2.csv'
        sep2 = ';'
        df2_expected = pd.DataFrame([["Ceci est un test", 1, "ceci est un test"],
                                     ["Ceci est un autre test; avec un point-virgule", 0, "ceci est un autre test avec un point virgule"]] * 15,
                                    columns=["x_col", "y_col", "texte_preprocess"])
        first_line2_expected = '#preprocess_P1'

        # Fonctionnement nominal
        df1, first_line1 = utils.read_csv(path1, sep=sep1, encoding='utf-8')
        df2, first_line2 = utils.read_csv(path2, sep=sep2, encoding='utf-8')
        self.assertEqual(df1.shape, df1_shape_expected)
        self.assertEqual(first_line1, first_line1_expected)
        pd.testing.assert_frame_equal(df2, df2_expected)
        self.assertEqual(first_line2, first_line2_expected)

        # Testing kwargs
        df3_expected = pd.DataFrame([["Ceci est un test", 1, "ceci est un test"],
                                     ["Ceci est un autre test; avec un point-virgule", 0, "ceci est un autre test avec un point virgule"]] * 2,
                                    columns=["x_col", "y_col", "texte_preprocess"])
        df3, first_line3 = utils.read_csv(path2, sep=sep2, encoding='utf-8', nrows=4)
        pd.testing.assert_frame_equal(df3, df3_expected)

        # Vérification des erreurs
        with self.assertRaises(ValueError):
            utils.read_csv('test_utils.py')
        with self.assertRaises(FileNotFoundError):
            utils.read_csv('toto.csv')


    @patch('logging.Logger._log')
    def test02_to_csv(self, PrintMockLog):
        '''Test de la fonction utils.to_csv'''
        # Data
        df = pd.DataFrame([['test', 'test'], ['toto', 'titi'], ['tata', 'tutu']], columns=['col1', 'col2'])
        fake_filepath = 'fake_csv.csv'
        # Clear
        if os.path.exists(fake_filepath):
            os.remove(fake_filepath)

        # Test nominal
        utils.to_csv(df, fake_filepath, first_line=None, sep=';', encoding='utf-8')
        self.assertTrue(os.path.exists(fake_filepath))
        reloaded_df = pd.read_csv(fake_filepath, sep=';', encoding='utf-8')
        pd.testing.assert_frame_equal(df, reloaded_df)

        # Avec first_line
        os.remove(fake_filepath)
        first_line = 'ligne1'
        utils.to_csv(df, fake_filepath, first_line=first_line, sep=';', encoding='utf-8')
        self.assertTrue(os.path.exists(fake_filepath))
        with open(fake_filepath, 'r', encoding='utf-8') as f:
            first_line_realoaded = f.readline().replace('\n', '').replace('\r', '')
        self.assertEqual(first_line, first_line_realoaded)
        reloaded_df2 = pd.read_csv(fake_filepath, sep=';', encoding='utf-8', skiprows=1)
        pd.testing.assert_frame_equal(df, reloaded_df2)

        # Clear
        if os.path.exists(fake_filepath):
            os.remove(fake_filepath)


    @patch('logging.Logger._log')
    def test03_display_shape(self, PrintMockLog):
        '''Test de la fonction utils.display_shape'''
        # On reenable le logger
        logging.disable(logging.NOTSET)

        # Vals à tester
        input_test = pd.DataFrame({'col1': ['A', 'A', 'D', np.nan, 'C', 'B'],
                                   'col2': [0, 0, 1, 2, 3, 1],
                                   'col3': [0, 0, 3, np.nan, np.nan, 3]})

        # Assert info called 1 time
        utils.display_shape(input_test)
        self.assertEqual(len(PrintMockLog.mock_calls), 1)

        # RESET DEFAULT
        logging.disable(logging.CRITICAL)


    def test04_get_chunk_limits(self):
        '''Test de la fonction utils.get_chunk_limits'''
        # Vals à tester
        input_test_1 = pd.DataFrame()
        expected_result_1 = [(0, 0)]
        input_test_2 = pd.DataFrame([0])
        expected_result_2 = [(0, 1)]
        input_test_3 = pd.DataFrame([0] * 100000)
        expected_result_3 = [(0, 10000), (10000, 20000), (20000, 30000), (30000, 40000),
                             (40000, 50000), (50000, 60000), (60000, 70000), (70000, 80000),
                             (80000, 90000), (90000, 100000)]
        input_test_4 = pd.DataFrame([0] * 100001)
        expected_result_4 = [(0, 10000), (10000, 20000), (20000, 30000), (30000, 40000),
                             (40000, 50000), (50000, 60000), (60000, 70000), (70000, 80000),
                             (80000, 90000), (90000, 100000), (100000, 100001)]
        input_test_5 = pd.DataFrame([0] * 100)
        expected_result_5 = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60),
                             (60, 70), (70, 80), (80, 90), (90, 100)]

        # Fonctionnement nominal
        self.assertEqual(utils.get_chunk_limits(input_test_1), expected_result_1)
        self.assertEqual(utils.get_chunk_limits(input_test_2), expected_result_2)
        self.assertEqual(utils.get_chunk_limits(input_test_3), expected_result_3)
        self.assertEqual(utils.get_chunk_limits(input_test_4), expected_result_4)
        self.assertEqual(utils.get_chunk_limits(input_test_5, chunksize=10), expected_result_5)

        # Vérification du type du/des input(s)
        with self.assertRaises(TypeError):
            utils.get_chunk_limits(3)
        with self.assertRaises(ValueError):
            utils.get_chunk_limits(input_test_1, chunksize=-1)


    def test05_get_configs_path(self):
        '''Test de la fonction utils.get_configs_path'''
        # Fonctionnement nominal
        path = utils.get_configs_path()
        self.assertEqual(os.path.isdir(path), True)
        self.assertEqual(path.endswith('configs'), True)


    def test06_get_data_path(self):
        '''Test de la fonction utils.get_data_path'''
        # Fonctionnement nominal
        path = utils.get_data_path()
        self.assertEqual(os.path.isdir(path), True)
        self.assertEqual(path.endswith('ynov-data'), True)

        # Avec un DIR_PATH != None
        current_dir = os.path.abspath(os.getcwd())
        utils.DIR_PATH = current_dir
        path = utils.get_data_path()
        self.assertEqual(os.path.isdir(path), True)
        self.assertEqual(path, os.path.join(current_dir, 'ynov-data'))

        # Nettoyage
        utils.DIR_PATH = None
        remove_dir(path)


    def test07_get_models_path(self):
        '''Test de la fonction utils.get_models_path'''
        # Fonctionnement nominal
        path = utils.get_models_path()
        self.assertEqual(os.path.isdir(path), True)
        self.assertEqual(path.endswith('ynov-models'), True)

        # Avec un DIR_PATH != None
        current_dir = os.path.abspath(os.getcwd())
        utils.DIR_PATH = current_dir
        path = utils.get_models_path()
        self.assertEqual(os.path.isdir(path), True)
        self.assertEqual(path, os.path.join(current_dir, 'ynov-models'))

        # Nettoyage
        utils.DIR_PATH = None
        remove_dir(path)


    def test08_get_pipelines_path(self):
        '''Test de la fonction utils.get_pipelines_path'''
        # Fonctionnement nominal
        path = utils.get_pipelines_path()
        self.assertEqual(os.path.isdir(path), True)
        self.assertEqual(path.endswith('ynov-pipelines'), True)

        # Avec un DIR_PATH != None
        current_dir = os.path.abspath(os.getcwd())
        utils.DIR_PATH = current_dir
        path = utils.get_pipelines_path()
        self.assertEqual(os.path.isdir(path), True)
        self.assertEqual(path, os.path.join(current_dir, 'ynov-pipelines'))

        # Nettoyage
        utils.DIR_PATH = None
        remove_dir(path)


    def test09_get_ressources_path(self):
        '''Test de la fonction utils.get_ressources_path'''
        # Fonctionnement nominal
        path = utils.get_ressources_path()
        self.assertEqual(os.path.isdir(path), True)
        self.assertEqual(path.endswith('ynov-ressources'), True)


    def test10_get_package_version(self):
        '''Test de la fonction utils.get_package_version'''
        # Fonctionnement nominal
        version = utils.get_package_version()
        self.assertEqual(type(version), str)


    def test11_flatten(self):
        '''Test de la fonction utils.flatten'''
        # Fonctionnement nominal
        test_list = [[1, 2], 3, [4], [1, [5, 8]]]
        expected_result = [1, 2, 3, 4, 1, 5, 8]
        flattened_list = list(utils.flatten(test_list))
        self.assertEqual(flattened_list, expected_result)


# Execution des tests
if __name__ == '__main__':
    # Start tests
    unittest.main()