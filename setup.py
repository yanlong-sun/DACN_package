from setuptools import setup

setup(
    name='bse_ml',
    version='1.0.1',
    author='Yanlong Sun, Anand Joshi',
    author_email='yanlongs@usc.edu',
    url='https://github.com/yanlong-sun/DACN_package',
    description='bse_ml',
    packages=['bse_ml'],
    install_requires=['tensorflow == 1.15',
                      'imageio',
                      'scikit-image'
                      ],
    package_data={'': ['*.m', '/modeldir/*.meta']},
    entry_points={
        'console_scripts': [
            'bse_ml=bse_ml:bse_ml'
        ]
    },
    include_package_data=True,
    zip_safe=False
)