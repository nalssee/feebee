from setuptools import setup

setup(name='feebee',
      version='0.1.16',
      description='datawork tools',
      url='https://github.com/nalssee/feebee.git',
      author='nalssee',
      author_email='kenjin@sdf.org',
      license='MIT',
      packages=['feebee'],
      # Install statsmodels manually using conda install
      # TODO: Not easy to install numpy and stuff without conda
      install_requires=[
          'sas7bdat==2.0.7',
          'psutil==5.4.3',
          'graphviz==0.8.2',
          'pandas'
      ],
      zip_safe=False)
