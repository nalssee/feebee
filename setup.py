from setuptools import setup

setup(name='feebee',
      version='0.1.0',
      description='datawork tools',
      url='https://github.com/nalssee/feebee.git',
      author='nalssee',
      author_email='kenjin@sdf.org',
      license='MIT',
      packages=['feebee'],
      # Install statsmodels manually with conda install
      install_requires=[
          'sas7bdat==2.0.7',
          'xlrd==1.1.0',
          'openpyxl==2.5.0',
          'psutil==5.4.3',
          'graphviz==0.8.2'
      ],
      # scripts=['bin/prepend', 'bin/fnguide', 'bin/xl2csv'],
      zip_safe=False)