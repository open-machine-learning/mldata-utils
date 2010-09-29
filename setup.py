from distutils.core import setup
import ml2h5

setup(
    name=ml2h5.PROGRAM_NAME,
    version=ml2h5.VERSION,
    description=ml2h5.DESCRIPTION,
    long_description=ml2h5.LONG_DESCRIPTION,
    maintainer=ml2h5.AUTHOR,
    maintainer_email=ml2h5.AUTHOR_EMAIL,
    license=ml2h5.LICENSE,
    url=ml2h5.URL,
    packages=['ml2h5', 'ml2h5.converter'],
    scripts=['scripts/ml2h5conv', 'scripts/ml2h5extract_data', 'scripts/ml2h5extract_task'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Natural Language :: English',
        'Operating System :: POSIX',
        'Programming Language :: Python',
        'Topic :: Utilities',
    ],
    platforms=['POSIX'],
)
