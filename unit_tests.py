import unittest
import pandas as pd
from graph import *

string_token_preprocessor = String_token_preprocessor()
bert_embedding_buffer = Bert_Embedding_Buffer()

class TestGraph(unittest.TestCase):
    def setUp(self):
        #called before every test
        self.df1 = pd.read_csv(r"unittest_data\testAB.csv")
        self.df2 = pd.read_csv(r"unittest_data\testBC.csv")
        self.df3 = pd.read_csv(r"unittest_data\fodors_zagats-master.csv")
    
    def tearDown(self):
        #called after every test
        pass

    def test_GraphConstructionBert(self):
        gl = Graph_list()
        g1 = Graph(self.df1, 'Table1', bert_embedding_buffer, string_token_preprocessor)
        self.assertEqual(g1.X.shape[0], 11)
        gl.add_item(g1)
        g2 = Graph(self.df2, 'Table2', bert_embedding_buffer, string_token_preprocessor)
        gl.add_item(g2)
        g2 = Graph(self.df2, 'Table2', bert_embedding_buffer, string_token_preprocessor)
        gl.add_item(g2)

        gl.save(r"unittest_data")
        gl2 = Graph_list(directory_name=r"unittest_data")

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

if __name__ == '__main__':
    unittest.main()