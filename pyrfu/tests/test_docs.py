#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import glob
import json
import os
import unittest

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2026"
__license__ = "MIT"
__version__ = "2.4.21"
__status__ = "Prototype"

# Example notebooks of the documentation (in a source checkout only)
DOCS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "docs", "examples"
)
NOTEBOOKS = sorted(glob.glob(os.path.join(DOCS_PATH, "*", "*.ipynb")))


@unittest.skipUnless(NOTEBOOKS, "documentation notebooks not found")
class DocsNotebooksTestCase(unittest.TestCase):
    # The notebooks are not executed by the docs build (nbsphinx_execute =
    # "never"): their saved outputs and headings are what the docs show

    @staticmethod
    def _load(path):
        with open(path, encoding="utf-8") as file:
            return json.load(file)

    def test_docs_notebooks_no_widget_outputs(self):
        # Figures saved as Jupyter widgets (%matplotlib widget, ipympl) are not
        # rendered by nbsphinx, and their gallery thumbnails are broken links
        for path in NOTEBOOKS:
            outputs = [
                o for c in self._load(path)["cells"] for o in c.get("outputs", [])
            ]
            widgets = [
                o
                for o in outputs
                if "application/vnd.jupyter.widget-view+json" in o.get("data", {})
            ]
            with self.subTest(notebook=os.path.basename(path)):
                self.assertListEqual(widgets, [])

    def test_docs_notebooks_inline_backend(self):
        # The notebooks that plot set the inline backend, so that they are not
        # saved with widget outputs from a Jupyter configured for ipympl
        for path in NOTEBOOKS:
            code = "".join(
                "".join(c["source"])
                for c in self._load(path)["cells"]
                if c["cell_type"] == "code"
            )
            if "matplotlib" in code:
                with self.subTest(notebook=os.path.basename(path)):
                    self.assertIn("%matplotlib inline", code)

    def test_docs_notebooks_title_without_math(self):
        # The first heading is the title of the gallery card, where math is not
        # rendered (the LaTeX source is shown)
        for path in NOTEBOOKS:
            cells = self._load(path)["cells"]
            title = "".join(cells[0]["source"]).splitlines()[0]
            with self.subTest(notebook=os.path.basename(path)):
                self.assertTrue(title.startswith("# "))
                self.assertNotIn("$", title)


if __name__ == "__main__":
    unittest.main()
